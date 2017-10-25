# SneakerNet
Retraining mobilenet for classifying sneakers, with Tensorflow.

## Model
Based heavily on the [Mobilenet V1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) and adapted directly from Tensorflow [retraining code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py)

Jointly predicts both the brand and name of sneaker, from a collection of web-scraped images.

## Data
Data scraping code uses a fork of [this code](https://github.com/hardikvasa/google-images-download)

Images of sneakers can be scraped automatically with `dataScraper.py` using the labels in `data/sneakers.txt`. A quick recursive call to `mogrify` will clean up the bulk of the invalid JPGs.

A first personal foray into Tensorflow, still in progress.
