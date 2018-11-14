import matplotlib.pyplot as plt
import numpy as np
import copy

import tensorflow as tf



tfe = tf.contrib.eager

tf.enable_eager_execution()

# Fetch and format the mnist data
(mnist_images, mnist_labels), (test_images,test_labels) = tf.keras.datasets.mnist.load_data()
n = mnist_labels.shape[0]
categorical = np.zeros((n, 10))
categorical[np.arange(n), mnist_labels] = 1
mnist_labels = categorical
n =test_labels.shape[0]
categorical = np.zeros((n, 10))
categorical[np.arange(n), test_labels] = 1
test_labels = categorical
dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255, tf.float32),
     tf.cast(mnist_labels, tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

test_dataset= tf.data.Dataset.from_tensor_slices(
    (tf.cast(test_images[..., tf.newaxis] / 255, tf.float32),
     tf.cast(test_labels, tf.int64)))
test_dataset = test_dataset.shuffle(1000).batch(32)

def loss_fn(x, y):
    logits = mnist_model(x, training=False)
    # Compute loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    return loss

def get_logit(x, class_ind):
    predictions = sub_model(x, training=False)
    return predictions[:, class_ind]


def get_grad(x):
    np_dtype = np.dtype('float32')

    logits = mnist_model(x, training=False)
    nb_classes = len(logits[0].numpy())

    nb_features = np.product(x.get_shape().as_list()[1:])

    # Compute the Jacobian components
    list_derivatives = []
    for class_ind in range(nb_classes):
        derivatives = tfe.gradients_function(get_logit)(x, class_ind)[0]
        list_derivatives.append(derivatives)
    grads = list_derivatives
    return grads

def fgm(x,
        model,
        y=None,
        eps=0.80,
        ord=np.inf,
        sanity_checks=True):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param logits: output of model.get_logits
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
   
    :return: a tensor for the adversarial example
    """
    logits = model(x, training=False)

    asserts = []


    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(logits, 1, keepdims=True)
        y = tf.to_float(tf.equal(logits, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keepdims=True)

    # Define gradient of loss wrt input
    loss = loss_fn(x, y)
    grad, = tfe.gradients_function(loss_fn)(x, y)[0]
    # print(grad)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(range(1, len(grad.get_shape())))
        avoid_zero_div = 1e-12
        print(grad.get_shape())

        avoid_nan_norm = tf.maximum(avoid_zero_div,
                                    tf.reduce_sum(tf.abs(grad),
                                                  reduction_indices=red_ind,
                                                  keepdims=True))
        normalized_grad = grad / avoid_nan_norm
    elif ord == 2:
        red_ind = list(range(1, len(grad.get_shape())))
        avoid_zero_div = 1e-12
        square = tf.maximum(avoid_zero_div,
                            tf.reduce_sum(tf.square(grad),
                                          reduction_indices=red_ind,
                                          keepdims=True))
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_grad = tf.multiply(eps, normalized_grad)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

    # If clipping is needed, reset all values outside of [clip_min, clip_max]

    if sanity_checks:
        with tf.control_dependencies(asserts):
            adv_x = tf.identity(adv_x)

    return adv_x


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
    for (batch, (images, labels)) in enumerate(dataset):
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
        epoch_accuracy(tf.argmax(mnist_model(images), axis=1, output_type=tf.int64), tf.argmax(labels, axis=1, output_type=tf.int64))

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
for (batch, (images, labels)) in enumerate(test_dataset.take(1)):
  logits = mnist_model(images, training=False)
  avg_loss(tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=logits, labels=labels)))
  accuracy(
      tf.argmax(logits, axis=1, output_type=tf.int64),
      tf.argmax(labels, axis=1, output_type=tf.int64))

print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
        (avg_loss.result(), 100 * accuracy.result()))


##############################################################################
#mnist_model now is a black box model, we only work with its predictions.
# now we want to train a sub model using https://arxiv.org/abs/1602.02697 method.


sub_model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

sub_loss_history = []

# keep results for plotting
sub_train_loss_results = []
sub_train_accuracy_results = []
sub_optimizer = tf.train.AdamOptimizer()
holdout = 150

x_sub = test_images[:holdout]
y_sub = test_labels[:holdout]
sub_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_sub[..., tf.newaxis] / 255, tf.float32),
     tf.cast(y_sub, tf.int64)))
sub_dataset = sub_dataset.shuffle(150).batch(32)

x_sub = tf.cast(x_sub[..., tf.newaxis] / 255, tf.float32)
y_sub=tf.cast(y_sub, tf.int64)
ITERATION = 3
LAMBDA=0.1
for i in range(ITERATION):
    #train sub
    num_epochs = 4
    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
        for (batch, (images, labels)) in enumerate(sub_dataset):
            with tf.GradientTape() as tape:
                logits = sub_model(images, training=True)
                loss_value = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

                loss_history.append(loss_value.numpy())
                grads = tape.gradient(loss_value, sub_model.variables)
                optimizer.apply_gradients(zip(grads, sub_model.variables),
                                          global_step=tf.train.get_or_create_global_step())
            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(sub_model(images), axis=1, output_type=tf.int64), tf.argmax(labels, axis=1, output_type=tf.int64))

        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

    #augmen x_sub
    x_sub_prev = copy.copy(x_sub)
    x_sub = np.vstack([x_sub_prev, x_sub_prev])
    y_sub_prev = copy.copy(y_sub)
    y_sub = np.vstack([y_sub_prev, y_sub_prev])

    if i < ITERATION -1:
        for (batch, (images, labels)) in enumerate(sub_dataset):
            new_x = copy.copy(images.numpy())
            new_y = copy.copy(labels.numpy())

            grads = get_grad(images)
            n=len(new_x)
            for j in range(n):
                new_x[j]= new_x[j]+ LAMBDA * grads[tf.argmax(labels[j], axis=0, output_type=tf.int64)][j]
            new_y = mnist_model(new_x, training=False)
            new_y = tf.argmax(new_y, axis=1, output_type=tf.int64)
            categorical = np.zeros((n, 10))
            categorical[np.arange(n), new_y] = 1
            y_sub = np.vstack([y_sub,categorical])
            x_sub = np.vstack([x_sub,new_x])

    sub_dataset = tf.data.Dataset.from_tensor_slices((x_sub,y_sub))
    sub_dataset = sub_dataset.shuffle(150).batch(32)

# now sub_model can be used to craft adverserial examples using fgm or jsma.