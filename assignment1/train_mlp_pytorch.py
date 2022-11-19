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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as torch
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn

MODEL_PATH = "model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"


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
    conf_mat = torch.zeros(size=(n_classes, n_classes)).to(device)

    for sample, label in zip(predictions, targets):
        prediction = torch.argmax(sample)
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
    confusion_matrix = confusion_matrix.to(device)
    metrics = {}
    metrics["accuracy"] = torch.sum(torch.diag(confusion_matrix)) / torch.sum(confusion_matrix)
    tp = torch.diag(confusion_matrix)
    fp = torch.sum(confusion_matrix, axis=0) - tp
    fn = torch.sum(confusion_matrix, axis=1) - tp

    metrics["precision"] = tp / (tp + fp)
    metrics["recall"] = tp / (tp + fn)
    metrics["f1_beta"] = (1 + beta ** 2) * metrics["precision"] * metrics["recall"] / (beta ** 2 * metrics["precision"] + metrics["recall"])
    return metrics


def evaluate_model(model, data_loader, num_classes=10, beta=1.):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.
    """
    model.eval()
    image_size = 32 * 32 * 3
    conf_matrix = torch.zeros(size=(num_classes, num_classes)).to(device)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            images = torch.reshape(images, shape=(images.shape[0], image_size))
            predictions = model(images)
            batch_conf_matrix = confusion_matrix(predictions, labels)
            conf_matrix += batch_conf_matrix

        metrics = confusion_matrix_to_metrics(conf_matrix, beta)
        metrics["confusion_matrix"] = deepcopy(conf_matrix)

    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir, save_model=True):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.
    """

    # Set the random seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    train_loader = cifar10_loader["train"]
    validation_loader = cifar10_loader["validation"]
    test_loader = cifar10_loader["test"]

    data_size = 32 * 32 * 3
    num_classes = 10

    model = MLP(data_size, hidden_dims, num_classes, use_batch_norm)
    model.to(device)
    loss_module = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr)

    train_accuracies = torch.zeros(epochs)
    val_accuracies = torch.zeros(epochs)

    train_losses = torch.zeros(epochs)
    val_losses = torch.zeros(epochs)

    best_accuracy = 0
    best_model = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = 0
        model.train()

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            images = torch.reshape(images, shape=(images.shape[0], data_size))

            opt.zero_grad()

            predictions = model(images)
            loss = loss_module(predictions, labels)

            loss.backward()
            opt.step()

            train_loss += loss.item()

        train_losses[epoch] = train_loss / len(train_loader)
        train_metrics = evaluate_model(model, train_loader)
        train_accuracies[epoch] = train_metrics["accuracy"]

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for images, labels in tqdm(validation_loader):
                images = images.to(device)
                labels = labels.to(device)
                images = torch.reshape(images, shape=(images.shape[0], data_size))

                predictions = model(images)
                loss = loss_module(predictions, labels)

                val_loss += loss.item()

        val_losses[epoch] = val_loss / len(validation_loader)
        val_metrics = evaluate_model(model, validation_loader)
        val_accuracies[epoch] = val_metrics["accuracy"]

        if best_accuracy < val_metrics["accuracy"]:
            best_accuracy = val_metrics["accuracy"]
            best_model = deepcopy(model)

    test_metrics = evaluate_model(best_model, test_loader)
    test_accuracy = test_metrics["accuracy"]

    logging_info = {
        "train_losses": train_losses,
        "validation_losses": val_losses,
        "train_accuracies": train_accuracies
    }

    if save_model:
        torch.save(best_model, MODEL_PATH)

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


def evaluate_fbeta_scores(model, data_dir, batch_size, num_classes=10):
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    test_loader = cifar10_loader["test"]
    BETAS = [.1, 1, 10]

    for beta in BETAS:
        metrics = evaluate_model(model, test_loader, num_classes, beta)
        f_beta_score = metrics["f1_beta"]
        print(f"F {beta} = {f_beta_score.tolist()}")

    print("Precision", metrics["precision"].tolist())
    print("Recall", metrics["recall"].tolist())


def plot_confusion_matrix(model, data_dir, batch_size, num_classes=10):
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    test_loader = cifar10_loader["test"]
    metrics = evaluate_model(model, test_loader, num_classes)
    confusion_matrix = metrics["confusion_matrix"]

    plt.figure(figsize = (10, 7))
    heatmap = seaborn.heatmap(confusion_matrix.long().cpu(), annot=True, fmt="g")
    figure = heatmap.get_figure()
    figure.savefig("confusion_matrix.png", dpi=400)


def find_best_lr(hidden_dims, batch_size, epochs, seed, data_dir):
    LEARNING_RATES = [10 ** lr for lr in range(-6, 2)]

    accuracies = torch.rand(size=(len(LEARNING_RATES), ))
    losses = torch.rand(size=(len(LEARNING_RATES), epochs))

    for i, lr in enumerate(LEARNING_RATES):
        _, val_accuracies, _, logging_info = train(hidden_dims, lr, False, batch_size, epochs, seed, data_dir, save_model=False)
        accuracy = torch.mean(val_accuracies)
        accuracies[i] = accuracy
        losses[i] = logging_info["validation_losses"]

    LEARNING_RATES = [str(lr) for lr in LEARNING_RATES]

    plt.bar(LEARNING_RATES, accuracies.tolist())
    plt.xlabel("Learning rate")
    plt.ylabel("Validation accuracy")
    plt.show()

    for loss, lr in zip(losses, LEARNING_RATES):
        plt.plot(loss, label=lr)
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

    plt.show()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

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

    if not os.path.exists(MODEL_PATH):
        model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
        print(test_accuracy.item())
        plot(val_accuracies, logging_info)
    else:
        model = torch.load(MODEL_PATH)
        model.to(device)
        evaluate_fbeta_scores(model, kwargs["data_dir"], kwargs["batch_size"])
        plot_confusion_matrix(model, kwargs["data_dir"], kwargs["batch_size"])
        find_best_lr(kwargs["hidden_dims"], kwargs["batch_size"], kwargs["epochs"], kwargs["seed"], kwargs["data_dir"])