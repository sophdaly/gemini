gemini 👯‍️
==============================

The purpose of this repo is to experiment with Siamese Networks and One Shot Learning, using FaceNet as inspiration and guidance.

Specifically, `gemini` fine tunes a pre-trained FaceNet model and evaluates performance when trained on regular and fine-grained LFW data. 


## References
* **TODO:** Add table to credit sources of pretrained model and datasets used

## Overview

### FaceNet
FaceNet is a facial recognition model.

* **TODO:** Add detail


### LFW
[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) is a standard dataset used for evaluating and benchmarking facial recognition algorithms. 

### Siamese Networks

Siamese networks are a special type of neural network architecture. Instead of a model learning to classify its inputs, the neural networks learns to differentiate between them. 


### Transfer Learning

I used a pretrained [FaceNet](https://github.com/davidsandberg/facenet) model from David Sandberg, to enrich images with 512 dimensional embedding.

Ideally I would like to restore pretrained model via Estimator API, and fine tune by appending additional layers. But, I couldn't find an easy way to do this, e.g. using Tensorflow Hub. So decided to keep FaceNet as a feature generation step, and train my own custom model separately to get started quickly.


The model is restored and images are enriched via the [`enrich.py`]() script.

## Getting Started

* **TODO:** add Makefile for downloading pretrained models


## Results


* **TODO:** add Makefile for downloading pretrained models

--------
