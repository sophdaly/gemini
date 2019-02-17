"""
Train model
"""

from gemini.utils.data.loading .input_data import input_data
from gemini.models.siamese_network.estimator import model_fn
import tensorflow as tf
import argparse
import json
import sys


FLAGS = None


def main(_):

    # Configure model params
    params = json.load(open(FLAGS.config))

    print("Training model: {}".format(params['model_dir']))

    config = tf.estimator.RunConfig(
        save_summary_steps = params['save_summary_steps'],
        save_checkpoints_steps=params['save_checkpoints_steps'],
        keep_checkpoint_max = params['keep_checkpoint_max']
    )

    # Create/Load model
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params['model_dir'],
        params=params,
        config=config
    )

    # Train model for specified number of epochs [default:-1 for infinite epochs]
    model.train(
        input_fn=lambda: input_data(
            data_path=params['train_data'],
            batch_size=params['batch_size'],
            repeat_count=params['num_epochs'],
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='gemini/models/configs/gemini_model.json',
                        help='Path to config file [default: %(default)s]')

    FLAGS, unparsed = parser.parse_known_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
