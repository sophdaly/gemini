"""
Learning ops
"""

import tensorflow as tf


def get_loss_op(anchor, positive, negative, margin):
    """
    FaceNet triplet loss
    """

    return []


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

