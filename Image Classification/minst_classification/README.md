# MNIST Classification with Neural Network

This repository contains a simple machine learning model for classifying handwritten digits from the MNIST dataset using a neural network built with TensorFlow/Keras.

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
```

## Setup
Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/mnist-classification.git
```


## Usage
Load the MNIST dataset, preprocess the data (normalize the images and convert labels to categorical format).
Build the neural network model using Keras:
Flatten the input images
Add a dense hidden layer with 128 units and ReLU activation
Add a softmax output layer with 10 units (for the 10 possible digit classes)
Compile the model with Adam optimizer and categorical cross-entropy loss function.
Train the model for 10 epochs and validate the model using a validation split of 20%.
Evaluate the trained model on the test set and print the test accuracy.
Save the trained model as mnist_model.h5.
Plot training and validation accuracy over epochs.

Run the script with:
```bash
python main.py
```
