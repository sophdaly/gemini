"""
Helper functions to prepare feature data
"""

from collections import Counter
import os


def import_lfw_data(data_path):
    """
    Import LFW dataset and process into (image, label) format
    """
    labels = []
    images = []

    # Dir names represent labels
    unique_labels = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    # Loop through directories importing images and labels
    for lab in unique_labels:
        path= os.path.join(data_path, lab)
        image_names = os.listdir(path)
        image_paths = [os.path.join(path, img) for img in image_names]

        for image in image_paths:
            images.append(image)
            labels.append(lab)

    assert len(images) == len(labels), "Error: Wtf your images and labels don't match?"

    labels_with_multiple_samples = [lab for lab, count in Counter(labels).items() if count > 1]

    print("Total labels: {}".format(len(unique_labels)))
    print("Total images: {}".format(len(images)))
    print("Total labels with more that one image samples: {}".format(len(labels_with_multiple_samples)))

    return images, labels
