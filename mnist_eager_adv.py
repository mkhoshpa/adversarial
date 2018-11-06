import matplotlib.pyplot as plt
import os
import time
import numpy as np
import random
import tensorflow as tf

# pylint: enable=g-bad-import-order


tfe = tf.contrib.eager

tf.enable_eager_execution()

# Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255, tf.float32),
     tf.cast(mnist_labels, tf.int64)))
dataset = dataset.shuffle(1000).batch(32)


def loss_fn(x, y):
    logits = mnist_model(x, training=False)
    # Compute loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    return loss


def fgm(x,
        model,
        y=None,
        eps=0.80,
        ord=np.inf,
        clip_min=None,
        clip_max=None,
        targeted=False,
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
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """
    logits = model(x, training=False)

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(utils_tf.assert_greater_equal(x, tf.cast(clip_min, x.dtype)))

    if clip_max is not None:
        asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

    # Make sure the caller has not passed probs by accident
    # assert logits.op.type != 'Softmax'

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
    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

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
    for (batch, (images, labels)) in enumerate(dataset.take(500)):
        with tf.GradientTape() as tape:
            logits = mnist_model(images, training=True)
            loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

            loss_history.append(loss_value.numpy())
            grads = tape.gradient(loss_value, mnist_model.variables)
            optimizer.apply_gradients(zip(grads, mnist_model.variables),
                                      global_step=tf.train.get_or_create_global_step())
        # Track progress
        epoch_loss_avg(loss_value)  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(mnist_model(images), axis=1, output_type=tf.int64), labels)

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
for (batch, (images, labels)) in enumerate(dataset.take(1)):
    t = random.randint(0, 31)
    logits = mnist_model((images[t:t + 1]).numpy(), training=False)
    print('predicted :')
    print(tf.argmax(logits, axis=1, output_type=tf.int64))
    print('label :')
    print(labels[t])
    adv_x = fgm(x=images[t:t + 1], model=mnist_model, ord=2)
    print('predicted adv:')
    logits = mnist_model(adv_x.numpy(), training=False)
    print(tf.argmax(logits, axis=1, output_type=tf.int64))
    # print(adv_x *255)
    adv_x = (adv_x.numpy()).reshape([28, 28])
    print(np.subtract(adv_x, images[t:t + 1].numpy().reshape([28, 28])))
    # plt.gray()
    # plt.imshow((images[t:t+1].numpy()).reshape([28,28]))
    plt.imshow(adv_x)