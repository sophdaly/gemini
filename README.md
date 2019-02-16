gemini 👯‍️
==============================

The purpose of this repo is to experiment with Siamese Networks and Facial Verification, using FaceNet as #inspo.

Specifically, `gemini` fine tunes a pre-trained [FaceNet](https://github.com/davidsandberg/facenet) model and evaluates performance when trained on regular and fine-grained LFW data. 


## References
* **TODO:** Add table to credit sources of pretrained model and datasets used

## Overview

### FaceNet
FaceNet ([FaceNet](https://arxiv.org/pdf/1503.03832.pdf), 2015) is a CNN model from Google researchers that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. 

### LFW
[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) (LFW) is the de-facto academic benchmarking dataset for face verification which contains more than 13,000 labelled facial images of 1680 people collected from the web. 

#### Fine-Grained LFW

[FGLFW](http://www.whdeng.cn/FGLFW/FGLFW.html) deliberately selects 3000 similarly-looking face pairs to replace the random negative pairs in LFW that are quite easy to distinguish. Its purpose is to close the large gap between the reported performance on benchmarks and performance on real world tasks.


### Siamese Networks

Siamese networks are a special type of neural network architecture. Instead of a model learning to classify its inputs, the neural networks learns to differentiate between them. 


### Transfer Learning

I used a pretrained [FaceNet](https://github.com/davidsandberg/facenet) model from David Sandberg, to enrich images with 512 dimensional embedding. 

Ideally I would like to restore pretrained model via Estimator API, and fine tune by appending additional layers. But, I couldn't find an easy way to do this, e.g. using Tensorflow Hub. So decided to keep FaceNet as a feature generation step, and train my own custom model separately to get started quickly.


The model is restored and images are enriched via the [`enrich.py`](https://github.com/sophdaly/gemini/features/enrich.py) script.

## Getting Started

* **TODO:** add Makefile for downloading pretrained models


## Results



--------
