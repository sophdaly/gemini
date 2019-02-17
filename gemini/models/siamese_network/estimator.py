"""
Model function
"""

from gemini.models.siamese_network.learning import get_train_op, get_loss_op
from gemini.models.siamese_network.inference import mlp
from typing import Dict, Any
import tensorflow as tf


def model_fn(features: Dict[str, tf.Tensor],
             labels: tf.Tensor,
             mode: tf.estimator.ModeKeys,
             params: Dict[str, Any]=None) -> tf.estimator.EstimatorSpec:
    """
    Estimator model function

    :param features: features tensor
    :param labels: labels tensor
    :param mode: train/evaluate/predict mode key implicitly passed through Estimator API
    :param params: dict of hyperparameters
    :param config: config object
    :return: EstimatorSpec
    """

    # -------
    # FEATURE GENERATION
    # -------


    # -------
    # SIAMESE NETWORK
    # -------
    with tf.variable_scope("siamese_network") as scope:
        with tf.name_scope("anchor_network"):
            anchor_feats = mlp(input_features=features['anchor'],
                             num_features=params['num_features'],
                             layer_config=params['layer_config'],
                             weight_decay=params['weight_decay'],
                             training_mode=mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope("pos_network"):
            pos_feats = mlp(input_features=features['positive'],
                              num_features=params['num_features'],
                              layer_config=params['layer_config'],
                              weight_decay=params['weight_decay'],
                              training_mode=mode == tf.estimator.ModeKeys.TRAIN,
                              reuse=True)

        with tf.name_scope("neg_network"):
            neg_feats = mlp(input_features=features['negative'],
                              num_features=params['num_features'],
                              layer_config=params['layer_config'],
                              weight_decay=params['weight_decay'],
                              training_mode=mode == tf.estimator.ModeKeys.TRAIN,
                              reuse=True)

    # -------
    # PREDICT
    # -------
    if mode == tf.estimator.ModeKeys.PREDICT:
        pass

    else:
        # -------
        # UPDATE GRAPH
        # -------

        # Determine triplets


        # Define loss

        # -------
        # TRAIN
        # -------
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            pass

        # -------
        # EVAL
        # -------
        elif mode == tf.contrib.learn.ModeKeys.EVAL:
            pass
