################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

import torch


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """
    n_classes = predictions.shape[1]
    conf_mat = np.zeros(shape=(n_classes, n_classes))

    for sample, label in zip(predictions, targets):
        prediction = np.argmax(sample)
        conf_mat[label][prediction] += 1

    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    metrics = {}
    metrics["accuracy"] = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    tp = np.diag(confusion_matrix)
    fp = np.sum(confusion_matrix, axis=0) - tp
    fn = np.sum(confusion_matrix, axis=1) - tp
    metrics["precision"] = tp / (tp + fp)
    metrics["recall"] = tp / (tp + fn)
    metrics["f1_beta"] = (1 + beta ** 2) * metrics["precision"] * metrics["recall"] / (beta ** 2 * metrics["precision"] + metrics["recall"])
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.
    """
    image_size = 32 * 32 * 3
    conf_matrix = np.zeros(shape=(num_classes, num_classes))

    for images, labels in data_loader:
        images = np.reshape(images, newshape=(images.shape[0], image_size))
        predictions = model.forward(images)
        batch_conf_matrix = confusion_matrix(predictions, labels)
        conf_matrix += batch_conf_matrix

    metrics = confusion_matrix_to_metrics(conf_matrix)

    return metrics


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    train_loader = cifar10_loader["train"]
    validation_loader = cifar10_loader["validation"]
    test_loader = cifar10_loader["test"]

    data_size = 32 * 32 * 3
    num_classes = 10

    model = MLP(data_size, hidden_dims, num_classes)
    loss_module = CrossEntropyModule()

    train_accuracies = np.zeros(epochs)
    val_accuracies = np.zeros(epochs)

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    best_accuracy = 0
    best_model = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = 0

        for images, labels in tqdm(train_loader):
            images = np.reshape(images, newshape=(images.shape[0], data_size))
            predictions = model.forward(images)
            loss = loss_module.forward(predictions, labels)

            dout = loss_module.backward(predictions, labels)
            model.backward(dout)

            for layer in model.layers:
                if hasattr(layer, "params"):
                    layer.params["weight"] -= lr * layer.grads["weight"]
                    layer.params["bias"] -= lr * layer.grads["bias"]

            train_loss += loss

        train_losses[epoch] = train_loss / len(train_loader)
        train_metrics = evaluate_model(model, train_loader)
        train_accuracies[epoch] = train_metrics["accuracy"]

        val_loss = 0
        for images, labels in tqdm(validation_loader):
            images = np.reshape(images, newshape=(images.shape[0], data_size))
            predictions = model.forward(images)
            loss = loss_module.forward(predictions, labels)
            val_loss += loss

        val_losses[epoch] = val_loss / len(validation_loader)
        val_metrics = evaluate_model(model, validation_loader)
        val_accuracies[epoch] = val_metrics["accuracy"]

        if best_accuracy < val_metrics["accuracy"]:
            best_accuracy = val_metrics["accuracy"]
            best_model = deepcopy(model)

        model.clear_cache()

    test_metrics = evaluate_model(best_model, test_loader)
    test_accuracy = test_metrics["accuracy"]

    logging_info = {
        "train_losses": train_losses,
        "validation_losses": val_losses,
        "train_accuracies": train_accuracies
    }

    return model, val_accuracies, test_accuracy, logging_info


def plot(val_accuracies, logging_info):
    train_losses = logging_info["train_losses"]
    val_losses = logging_info["validation_losses"]
    train_accuracies = logging_info["train_accuracies"]

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Epoch Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.title("Model Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
    print(test_accuracy)
    plot(val_accuracies, logging_info)