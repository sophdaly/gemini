"""
MLP
"""

import tensorflow as tf


def mlp(input_features, num_features, layer_config, weight_decay=0.0, training_mode=True, reuse=False):
    """
    Configurable Multilayer Perceptron network with L2 normalisation on final layer

    - Number of layers, size of layers and optional dropout rates specified in layer_config
    - Optionally apply weight decay cost to loss function by regularizing weight matrix at each layer
    - Optionally reuse variables for second sister of siamese network
    """

    layers = layer_config['layer_sizes']
    dropout_rates = layer_config['dropout_rates']

    out = tf.reshape(input_features, [-1, num_features])

    for idx, (layer_size, dropout_rate) in enumerate(zip(layers, dropout_rates)):

        out = tf.layers.dense(
            inputs=out,
            units=layer_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
            bias_initializer=tf.constant_initializer(0.1),
            name="fc_{}".format(idx),
            reuse=reuse
        )

        out = tf.layers.dropout(inputs=out, rate=dropout_rate, training=training_mode)

    # L2 normalisation
    normed_features = tf.nn.l2_normalize(out, axis=1, name="l2_norm")

    return normed_features
