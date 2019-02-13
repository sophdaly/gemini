"""
Load pretrained FaceNet model for feature generation

Reference: https://github.com/davidsandberg/facenet
"""

import tensorflow as tf
from scipy import misc
from math import ceil
import numpy as np


class FaceNet(object):

    def __init__(self, sess, model_path, verbose=False):
        """
        Load pretrained FaceNet model from model_path
        """
        self.sess = sess
        self.graph = self._load_model(model_path, verbose)

        # Access restored placeholder variables to feed new data
        self.input_pl = self.graph.get_tensor_by_name("input:0")
        self.phase_train_pl = self.graph.get_tensor_by_name("phase_train:0")

        # Access restored embedding tensor to re run
        self.embeddings = self.graph.get_tensor_by_name("embeddings:0")

        # Standard input image size for FaceNet
        self.image_size = 160

    def enrich(self, images, batch_size):
        """
        Enrich input images with feature embeddings from FaceNet model
        Return 512 dimensional embedding array for each image
        """
        # Run forward pass to calculate embeddings
        num_images = len(images)
        num_batches = int(ceil(num_images / batch_size))

        # Prepare embedding array to hold embeddings
        embedding_size = self.embeddings.get_shape()[1]
        embedding_array = np.zeros((num_images, embedding_size))

        # Pass images through forward pass in batches
        for i in range(num_batches):

            # Keep track of how many images have been processed
            n = num_images if i == (num_batches - 1) else (i + 1) * batch_size

            # Forward pass batch through FaceNet to compute embeddings
            batch_embeddings = self._forward_pass(batch=images[i * batch_size:n])

            # Add embeddings to array
            embedding_array[i * batch_size:n, :] = batch_embeddings

        return embedding_array.tolist()

    def _load_model(self, model_path, verbose):
        """
        Load pretrained FaceNet model from file
        """
        # Get latest checkpoint file from dir
        latest_checkpoint = tf.train.latest_checkpoint(model_path)

        # Load latest checkpoint Graph via import_meta_graph:
        #   - construct protocol buffer from file content
        #   - add all nodes to current graph and recreate collections
        #   - return Saver
        saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
        saver.restore(self.sess, latest_checkpoint)

        # Retrieve protobuf graph definition
        graph = tf.get_default_graph()

        # Optionally print restored operations, variables and saved values
        if verbose:
            print("Restored Operations from MetaGraph:")
            for op in graph.get_operations():
                print(op.name)
            print("Restored Variables from MetaGraph:")
            for var in tf.global_variables():
                print(var.name)
            print("Restored Saved Variables from Checkpoint:")
            for init_var in tf.global_variables():
                try:
                    print("{}: {}".format(init_var.name, init_var.eval()))
                except Exception:
                    pass

        print("Restoring Model: {}".format(model_path))

        return graph

    def _forward_pass(self, batch):
        """
        Forward pass batch of images through FaceNet model
        """
        # Load images
        images = load_images(image_paths=batch, image_size=self.image_size)

        feed_dict = {
            self.input_pl: images,
            self.phase_train_pl: False
        }

        # Eval embeddings tensor with input feed dict
        batch_embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)

        return batch_embeddings


def load_images(image_paths, image_size, normalise=True):
    """
    Load image data from paths into correct image format and implement optional normalising preprocessing step
    Removed cropping and flipping preprocessing step
    """
    num_samples = len(image_paths)
    images = np.zeros((num_samples, image_size, image_size, 3))

    # Loop through images
    for i in range(num_samples):
        img = misc.imread(image_paths[i])

        # Resize
        img = misc.imresize(img, (image_size, image_size), interp='bilinear')

        if img.ndim == 2:
            img = _to_rgb(img)

        if normalise:
            img = _normalise(img)

        images[i,:,:,:] = img

    return images


def _normalise(x):
    """
    Normalise images aka FaceNet's 'prewhiten' preprocessing step
    """
    std_adj = np.maximum(np.std(x), 1.0 / np.sqrt(x.size))
    return np.multiply(np.subtract(x, np.mean(x)), 1 / std_adj)


def _to_rgb(img):
    """
    Convert image to RGB color space
    """
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img

    return ret
