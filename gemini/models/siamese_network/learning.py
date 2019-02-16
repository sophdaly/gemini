"""
Learning ops
"""

import tensorflow as tf


def get_loss_op(logits, labels, weights=None, label_smoothing=0.0, weigh_positive_labels_only=False):
    """
    Sigmoid cross entropy with per task weighting

    - Apply sigmoid activation and compute weighted cross entropy loss to each index of input logits s.t loss for
    majority classes is down weighted and minority classes up weighted

    - Optionally only apply weights to positive labels

    - Optionally apply label_smoothing, nonzero value smooth the labels towards 1/2
    """
    # Cast one hot labels to floats
    float_labels = tf.cast(labels, dtype=tf.float32)

    # Apply loss weights
    if weigh_positive_labels_only:
        weights = 1.0 if weights is None else (float_labels * (weights - 1.0) + 1.0)
    else:
        weights = 1.0 if weights is None else tf.reshape(weights, (1, len(weights)))

    loss_op = tf.losses.sigmoid_cross_entropy(
        logits=logits,
        multi_class_labels=float_labels,
        weights=weights,
        label_smoothing=label_smoothing
    )

    return loss_op


def get_train_op(loss_op, learning_rate):
    """
    Create an optimizer and apply to all trainable variables
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss_op,
        name='train_op',
        global_step=tf.train.get_global_step()
    )

    return train_op


def l2_distance(x, y):
    """
    Euclidean distance between vectors
    """
    # Add a small value 1e-6 to increase the stability of calculating the gradients for sqrt
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1) + 1e-6)

