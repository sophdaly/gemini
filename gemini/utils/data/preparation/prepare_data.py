"""
Script for preparing positive matches for training and testing
Negative matches to be appended at train time
"""

import pandas as pd
import itertools
import argparse
import time
import os


def _prepare_positive_matches_df(features_df):
    """
    Create dataframe of positive matches
    """
    anchor_embs, positive_embs, anchor_images, positive_images, labels = [], [], [], [], []

    # Group samples by label and create pairwise entries for learning
    grouped = features_df.groupby('label')

    for label, group in grouped:

        # Get all images for specific label
        images = list(group.image)

        # TODO: maybe rethink below logic later for creating negative samples
        if len(images) > 1:

            # Get all pairwise combinations of images
            for anc_img, pos_img in itertools.combinations(images, 2):

                # This is probably excessively expensive 🙃
                anc_emb = group.loc[group.image == anc_img, 'embedding'].item()
                pos_emb = group.loc[group.image == pos_img, 'embedding'].item()

                # Update tracker arrays
                labels.append(label)
                anchor_images.append(anc_img)
                anchor_embs.append(anc_emb)
                positive_images.append(pos_img)
                positive_embs.append(pos_emb)


    # Create dataframe
    matches_df = pd.DataFrame.from_dict(
        {
            'label': labels,
            'anchor': anchor_embs,
            'positive': positive_embs,
            'anchor_image': anchor_images,
            'positive_image': positive_images
        }
    )

    return matches_df


def _prepare_data(input_path, output_path, train_test_ratio):
    """
    Input features data and prepare positive matches and output train/test data for learning
    """
    # Read in features dataframe
    features_df = pd.read_csv(input_path)

    # Prepare positive matches df
    matches_df = _prepare_positive_matches_df(features_df=features_df)

    print("Total Labels: {}".format(len(set(matches_df.label))))
    print("Total Anchor <> Positive pairs: {}".format(len(matches_df)))

    train_df = matches_df.sample(frac=train_test_ratio, random_state=200)
    test_df = matches_df.drop(train_df.index)

    print("Total TRAIN samples: {} [{:.2f} %]".format(len(train_df), len(train_df)/len(matches_df) * 100.0))
    print("Total TEST samples: {} [{:.2f} %]".format(len(test_df), len(test_df)/len(matches_df) * 100.0))

    # Output to file
    train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)
    matches_df.to_csv(os.path.join(output_path, 'total.csv'), index=False)

    print("Data written to file: {}".format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_file', type=str, default='data/processed/feature_data.csv',
                        help='Path to input features file')
    parser.add_argument('--output_path', type=str, default='data/processed/',
                        help='Path to output feature data directory')
    parser.add_argument('-t', '--train-test-ratio', type=float,
                        help="train:test data split ratio (default: %(default)s)",
                        default=0.8)

    args = parser.parse_args()

    t0 = time.time()
    _prepare_data(input_path=args.features_file,
                  output_path=args.output_path,
                  train_test_ratio=args.train_test_ratio)

    print("Time Taken: {} s".format(time.time() - t0))