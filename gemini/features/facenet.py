"""
Load pretrained FaceNet model for feature generation

Reference: https://github.com/davidsandberg/facenet
"""

import tensorflow as tf

class FaceNet(object):

    def __init__(self, sess, model_path, verbose=False):
        """
        Load pretrained FaceNet model from model_path
        """
        self.sess = sess
        self.graph = self._load_model(model_path, verbose)

    def enrich(self, images):
        """
        Enrich input images with feature embeddings from FaceNet model
        """
        # Access restored placeholder variables to feed new data
        input_pl = self.graph.get_tensor_by_name("input:0")
        phase_train_pl = self.graph.get_tensor_by_name("phase_train:0")

        # Access restored embedding tensor to re run
        embeddings = self.graph.get_tensor_by_name("embeddings:0")

        feed_dict = {
            input_pl: images,
            phase_train_pl: False
        }

        print("Enriching images with FaceNet embeddings")
        enriched_images = self.sess.run(embeddings, feed_dict=feed_dict)

        return enriched_images

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
