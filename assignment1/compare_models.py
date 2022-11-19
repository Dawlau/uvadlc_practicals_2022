################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.
    """
    hyperparameters = [
        {
            "n_hidden": [128],
            "epochs": 20,
            "lr": 0.1
        },
        {
            "n_hidden": [256, 128],
            "epochs": 20,
            "lr": 0.1
        },
        {
            "n_hidden": [512, 256, 128],
            "epochs": 20,
            "lr": 0.1
        }
    ]
    results = []

    for i, settings in enumerate(hyperparameters):
        n_hiddens = settings["n_hidden"]
        lr = settings["lr"]
        epochs = settings["epochs"]

        _, val_accuracies, _, logging_info = train_mlp_pytorch.train(n_hiddens, lr, False, 128, epochs, 42, "data/", save_model=False)

        train_accuracies = logging_info["train_accuracies"]
        results.append([val_accuracies, train_accuracies])

    torch.save(results, results_filename)


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    """
    results = torch.load(results_filename)

    fig, axis = plt.subplots(2, 3)

    for i, (val_accuracies, train_accuracies) in enumerate(results):
        axis[0][i].plot(train_accuracies, label="Train Accuracy")
        axis[0][i].legend()
        axis[0][i].set_xlabel("Epochs")
        axis[0][i].set_ylabel("Accuracy")

        axis[1][i].plot(val_accuracies, label="Valid Accuracy")
        axis[1][i].legend()
        axis[1][i].set_xlabel("Epochs")
        axis[1][i].set_ylabel("Accuracy")

    plt.show()


if __name__ == '__main__':
    FILENAME = 'results.txt'
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)