"""
Learning ops
"""

import tensorflow as tf


def get_loss_op(anchor, positive, negative, margin):
    """
    FaceNet triplet loss
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = l2_distance(x=anchor, y=positive)
        neg_dist = l2_distance(x=anchor, y=negative)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


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

