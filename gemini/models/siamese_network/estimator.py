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
    # SIAMESE NETWORK
    # -------
    embeddings = mlp(input_features=features['embedding'],
                     num_features=params['num_features'],
                     layer_config=params['layer_config'],
                     weight_decay=params['weight_decay'],
                     training_mode=mode == tf.estimator.ModeKeys.TRAIN)

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
