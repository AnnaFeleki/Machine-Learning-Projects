# MNIST Classification with Neural Network

This repository contains a simple machine learning model for classifying handwritten digits from the MNIST dataset using a neural network built with TensorFlow/Keras.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

This project demonstrates how to classify images of handwritten digits from the MNIST dataset using a simple neural network. The model is built using Keras with TensorFlow as the backend.

The dataset consists of 60,000 training images and 10,000 test images of 28x28 grayscale digits (0-9). The neural network model consists of:
- A flattening layer to convert the 2D images into 1D vectors
- A hidden dense layer with ReLU activation
- A softmax output layer to classify the digits into 10 classes

## Requirements

To run this project, you will need to have the following Python libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `keras`

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib tensorflow keras
