import matplotlib.pyplot as plt
import os
import time
import numpy as np
import random
import tensorflow as tf
import copy

# pylint: enable=g-bad-import-order


tfe = tf.contrib.eager

tf.enable_eager_execution()
np_dtype = np.dtype('float32')

# Fetch and format the mnist data
(mnist_images, mnist_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
n = mnist_labels.shape[0]
categorical = np.zeros((n, 10))
categorical[np.arange(n), mnist_labels] = 1
mnist_labels = categorical
n = test_labels.shape[0]
categorical = np.zeros((n, 10))
categorical[np.arange(n), test_labels] = 1
test_labels = categorical

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255, tf.float32),
     tf.cast(mnist_labels, tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(test_images[..., tf.newaxis] / 255, tf.float32),
     tf.cast(test_labels, tf.int64)))
test_dataset = test_dataset.shuffle(1000).batch(32)


def loss_fn(x, y):
    logits = mnist_model(x, training=False)
    # Compute loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    return loss


def get_logit(x, class_ind):
    predictions = mnist_model(x, training=False)
    return predictions[:, class_ind]


def saliency_map(grads_target, grads_other, search_domain, increase):
    """
    TensorFlow implementation for computing saliency maps
    :param grads_target: a matrix containing forward derivatives for the
                         target class
    :param grads_other: a matrix where every element is the sum of forward
                        derivatives over all non-target classes at that index
    :param search_domain: the set of input indices that we are considering
    :param increase: boolean; true if we are increasing pixels, false otherwise
    :return: (i, j, search_domain) the two input indices selected and the
             updated search domain
    """
    # Compute the size of the input (the number of features)
    nf = len(grads_target)

    # Remove the already-used input features from the search space
    invalid = list(set(range(nf)) - search_domain)
    increase_coef = (2 * int(increase) - 1)
    grads_target[invalid] = -increase_coef * np.max(np.abs(grads_target))
    grads_other[invalid] = increase_coef * np.max(np.abs(grads_other))

    # Create a 2D numpy array of the sum of grads_target and grads_other
    target_sum = grads_target.reshape((1, nf)) + grads_target.reshape((nf, 1))
    other_sum = grads_other.reshape((1, nf)) + grads_other.reshape((nf, 1))

    # Create a mask to only keep features that match saliency map conditions
    if increase:
        scores_mask = ((target_sum > 0) & (other_sum < 0))
    else:
        scores_mask = ((target_sum < 0) & (other_sum > 0))

    # Create a 2D numpy array of the scores for each pair of candidate features
    scores = scores_mask * (-target_sum * other_sum)

    # A pixel can only be selected (and changed) once
    np.fill_diagonal(scores, 0)

    # Extract the best two pixels
    best = np.argmax(scores)
    p1, p2 = best % nf, best // nf

    # Remove used pixels from our search domain
    search_domain.discard(p1)
    search_domain.discard(p2)

    return p1, p2, search_domain


def jsma(sample,
         target,
         theta,
         gamma,
         clip_min,
         clip_max):
    """
    TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
    for details about the algorithm design choices).
    :param x: the input placeholder
    :param predictions: the model's symbolic output (the attack expects the
                  probabilities, i.e., the output of the softmax, but will
                  also work with logits typically)
    :param grads: symbolic gradients
    :param sample: numpy array with sample input
    :param target: target class for sample input
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param clip_min: minimum value for components of the example returned
    :param clip_max: maximum value for components of the example returned
    :return: an adversarial sample
    """

    # Copy the source sample and define the maximum number of features
    # (i.e. the maximum number of iterations) that we may perturb
    adv_x = copy.copy(sample)
    # count the number of features. For MNIST, 1x28x28 = 784; for
    # CIFAR, 3x32x32 = 3072; etc.
    nb_features = np.product(adv_x.get_shape().as_list()[1:])
    print('nb_features')
    print(nb_features)
    # reshape sample for sake of standardization
    original_shape = adv_x.shape
    adv_x = np.reshape(adv_x, (1, nb_features))
    # compute maximum number of iterations
    print(gamma)
    print(nb_features * gamma)
    max_iters = np.floor(nb_features * gamma / 2)

    # Find number of classes based on grads

    increase = bool(theta > 0)

    # Compute our initial search domain. We optimize the initial search domain
    # by removing all features that are already at their maximum values (if
    # increasing input features---otherwise, at their minimum value).
    if increase:
        search_domain = {i for i in range(nb_features) if adv_x[0, i] < clip_max}
    else:
        search_domain = {i for i in range(nb_features) if adv_x[0, i] > clip_min}

    # Initialize the loop variables
    iteration = 0
    adv_x_original_shape = np.reshape(adv_x, original_shape)
    # current = utils_tf.model_argmax(sess, x, predictions, adv_x_original_shape, feed=feed)

    logits = mnist_model(adv_x_original_shape, training=False)
    nb_classes = len(logits[0].numpy())
    print('nb_classes')
    print(logits[0].numpy())
    if adv_x_original_shape.shape[0] == 1:
        current = np.argmax(logits)
    else:
        current = np.argmax(logits, axis=1)

    print("Starting JSMA attack up to {} iterations".format(max_iters))
    # Repeat this main loop until we have achieved misclassification
    while (current != target and iteration < max_iters
           and len(search_domain) > 1):
        # Reshape the adversarial example
        adv_x_original_shape = np.reshape(adv_x, original_shape)

        # Compute the Jacobian components
        list_derivatives = []
        jacobian_val = np.zeros((nb_classes, nb_features), dtype=np_dtype)
        for class_ind in range(nb_classes):
            derivatives, = tfe.gradients_function(get_logit)(adv_x_original_shape, class_ind)[0]
            list_derivatives.append(derivatives)
        grads = list_derivatives
        for class_ind, grad in enumerate(grads):
            jacobian_val[class_ind] = np.reshape(grad, (1, nb_features))

            # Sum over all classes different from the target class to prepare for
            # saliency map computation in the next step of the attack
        print(nb_classes)
        print(target)
        other_classes = list(range(nb_classes))
        other_classes.remove(target)
        grads_others = np.sum(jacobian_val[other_classes, :], axis=0)
        grads_target = jacobian_val[target]
        if iteration % ((max_iters + 1) // 5) == 0 and iteration > 0:
            print("Iteration {} of {}".format(iteration,
                                              int(max_iters)))
        # Compute the saliency map for each of our target classes
        # and return the two best candidate features for perturbation
        i, j, search_domain = saliency_map(grads_target, grads_others,
                                           search_domain, increase)

        # Apply the perturbation to the two input features selected previously
        if increase:
            adv_x[0, i] = np.minimum(clip_max, adv_x[0, i] + theta)
            adv_x[0, j] = np.minimum(clip_max, adv_x[0, j] + theta)
        else:
            adv_x[0, i] = np.maximum(clip_min, adv_x[0, i] - theta)
            adv_x[0, j] = np.maximum(clip_min, adv_x[0, j] - theta)

        # Update our current prediction by querying the model
        logits = mnist_model(adv_x_original_shape, training=False)

        if adv_x_original_shape.shape[0] == 1:
            current = np.argmax(logits)
        else:
            current = np.argmax(logits, axis=1)

        # Update loop variables
        iteration = iteration + 1

    if current == target:
        print("Attack succeeded using {} iterations".format(iteration))
    else:
        print(("Failed to find adversarial example " +
               "after {} iterations").format(iteration))

    # Compute the ratio of pixels perturbed by the algorithm
    percent_perturbed = float(iteration * 2) / nb_features

    # Report success when the adversarial example is misclassified in the
    # target class
    if current == target:
        return np.reshape(adv_x, original_shape), 1, percent_perturbed
    else:
        return np.reshape(adv_x, original_shape), 0, percent_perturbed


# Build the model
input_shape = (28, 28, 1)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], padding='same', activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(32, [3, 3], padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, [3, 3], padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], padding='same', activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10)
])

optimizer = tf.train.AdamOptimizer()

loss_history = []

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 1
for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    for (batch, (images, labels)) in enumerate(dataset.take(400)):
        with tf.GradientTape() as tape:
            logits = mnist_model(images, training=True)
            loss_value = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

            loss_history.append(loss_value.numpy())
            grads = tape.gradient(loss_value, mnist_model.variables)
            optimizer.apply_gradients(zip(grads, mnist_model.variables),
                                      global_step=tf.train.get_or_create_global_step())
        # Track progress
        epoch_loss_avg(loss_value)  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(mnist_model(images), axis=1, output_type=tf.int64),
                       tf.argmax(labels, axis=1, output_type=tf.int64))

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

# end epoch
train_loss_results.append(epoch_loss_avg.result())
train_accuracy_results.append(epoch_accuracy.result())
print(mnist_model.summary())
# print(loss_history)
plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()

avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
accuracy = tfe.metrics.Accuracy()
for (batch, (images, labels)) in enumerate(test_dataset):
    logits = mnist_model(images, training=False)
    avg_loss(tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels)))

    accuracy(
        tf.argmax(logits, axis=1, output_type=tf.int64),
        tf.argmax(labels, axis=1, output_type=tf.int64))
print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
      (avg_loss.result(), 100 * accuracy.result()))

for (batch, (images, labels)) in enumerate(dataset.take(1)):
    t = random.randint(0, 31)
    logits = mnist_model((images[t:t + 1]).numpy(), training=False)
    print('predicted :')
    print(tf.argmax(logits, axis=1, output_type=tf.int64))
    print('label :')
    print(labels[t])
    adv_x, success, percent = jsma(images[t:t + 1], 6,
                                   1.,
                                   .1,
                                   0.,
                                   1.)
    print('predicted adv:')
    logits = mnist_model(adv_x, training=False)
    print(tf.argmax(logits, axis=1, output_type=tf.int64))
    print('success ')
    print(success)
    print('percent ')
    print(percent)
    # print(adv_x *255)
    adv_x = (adv_x).reshape([28, 28])
    # print(np.subtract(adv_x, images[t:t + 1].numpy().reshape([28, 28])))
    # plt.gray()
    # plt.imshow((images[t:t+1].numpy()).reshape([28,28]))
    plt.imshow(adv_x)
