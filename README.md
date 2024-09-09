# Handwritten Digit Recognition using Neural Networks

This repository contains a Python implementation using PyTorch to classify handwritten digits from the MNIST dataset. The neural network architecture consists of three layers: an input layer with 784 nodes, a hidden layer with 30 nodes, and an output layer with 10 nodes for digit classification using sigmoid normalization and Cross Entropy Loss for optimization.

## Project Overview

The objective of this project is to accurately classify handwritten digits (0-9) using a neural network trained on the popular MNIST dataset.

## Project Structure

- **NeuralNetwork_Handwritten_Digit_Recognition.ipynb**: Jupyter notebook containing the Python code for the neural network implementation.
- **data/**: Directory containing the MNIST dataset used for training and testing the model.

## Dataset Details

The MNIST dataset consists of grayscale images of handwritten digits, split into a training set and a testing set. The training set is used for model training, while the testing set evaluates its performance.

## Model Architecture

- **Input Layer**: 784 nodes representing a flattened 28x28 pixel image.
- **Hidden Layer**: 30 nodes with ReLU activation function.
- **Output Layer**: 10 nodes with softmax activation. The digit with the highest output probability is predicted.

## Training Process

The model is trained using stochastic gradient descent with Cross Entropy Loss as the optimization metric. Training involves multiple epochs to minimize loss and improve accuracy.

## Performance Metrics

After training, the model achieved an accuracy of 95.3% on the test dataset, indicating the percentage of correct predictions.
