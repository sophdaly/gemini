"""
Helper functions for loading features into a TFRecordDataset
"""

from ast import literal_eval
import tensorflow as tf
import pandas as pd


BUFFER_SIZE = 10000
FACENET_EMBEDDING_DIM = 512


def input_data(data_path, batch_size, repeat_count, shuffle=True):
    """
    Prepare dataset for train/test data repeating data repeat_count times and batching
    """
    dataset = _prepare_dataset(data_path=data_path,
                               batch_size=batch_size,
                               repeat_count=repeat_count,
                               shuffle=shuffle)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def _prepare_dataset(data_path, batch_size, repeat_count, shuffle):
    """
    Input data to Dataset and optionally shuffle
    """
    dataset = _load_dataset(data_path, embedding_dim=FACENET_EMBEDDING_DIM)

    # Optionally reshuffle data each time it is iterated over (disable when predicting locally)
    # For completely uniform shuffling set buffer_size parameter to the number of elements in the dataset
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=BUFFER_SIZE,
            reshuffle_each_iteration=True
        )

    # Set count to repeat dataset to - default is forever and ever #infinity
    repeated_data = dataset.repeat(count=repeat_count)

    # Batch that
    batched = repeated_data.batch(batch_size)

    return batched


def _load_dataset(data_path, embedding_dim):
    """
    Load data from csv
    """
    df = pd.read_csv(data_path)

    # Evaluate embedding array field
    df.embedding = df.embedding.apply(literal_eval)

    # Generate meta dict of types and shapes for generator
    gen_types, gen_shapes = _generator_meta(embedding_dim)

    def csv_generator():
        for idx, rows in df.iterrows():
            yield rows.to_dict()

    dataset = tf.data.Dataset().from_generator(
        csv_generator,
        output_types=gen_types,
        output_shapes=gen_shapes
    )

    return dataset.map(_format_input_features)


def _format_input_features(features_dict):
    """
    Just make sure all features are cast to correct type
    Return features dict and label
    """
    features = {
        'image': tf.cast(features_dict['image'], tf.string),
        'embedding': tf.cast(features_dict['embedding'], tf.float32),
    }

    label = tf.cast(features_dict['label'], tf.string)

    return features, label


def _generator_meta(embedding_dim):
    dict_types = {
        'image': tf.string,
        'label': tf.string,
        'embedding': tf.float32
    }
    dict_shapes = {
        'image': (),
        'label': (),
        'embedding': (embedding_dim),
    }

    return dict_types, dict_shapes
