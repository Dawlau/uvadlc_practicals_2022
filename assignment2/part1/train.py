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
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from tqdm import tqdm
from copy import deepcopy

from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    for parameter in model.parameters():
        parameter.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    torch.nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    torch.nn.init.zeros_(model.fc.bias)
    model.fc.requires_grad_ = True

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(data_dir, augmentation_name=augmentation_name)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with validation after each epoch. Save the best model.
    model.to(device)
    loss_module = nn.CrossEntropyLoss()

    best_accuracy = 0
    best_model = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        model.train()

        print("Training...")
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            predictions = model(images)
            loss = loss_module(predictions, labels)

            loss.backward()
            optimizer.step()

        all_predictions = torch.tensor([]).to(device)
        print("Evaluating on validation dataset...")
        with torch.no_grad():
            model.eval()
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                loss = loss_module(predictions, labels)

                predictions = torch.argmax(predictions, dim=1)
                all_predictions = torch.cat([all_predictions, predictions == labels], axis=0)

        val_accuracy = all_predictions.sum().item() / all_predictions.shape[0]
        print(f"Validation Accuracy: {val_accuracy}")

        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy
            best_model = deepcopy(model)

    torch.save(best_model, checkpoint_name)

    # Load the best model on val accuracy and return it.
    best_model = torch.load(checkpoint_name)

    return best_model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()
    model.to(device)

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().

    with torch.no_grad():
        all_predictions = torch.tensor([]).to(device)
        print("Evaluating on test dataset...")
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            predictions = torch.argmax(predictions, dim=1)
            all_predictions = torch.cat([all_predictions, predictions == labels], axis=0)

        accuracy = all_predictions.sum().item() / all_predictions.shape[0]

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, add_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = get_model()

    # Train the model
    CHECKPOINT_NAME = "model.pt"
    model = train_model(model, lr, batch_size, epochs, data_dir, CHECKPOINT_NAME, device, augmentation_name)

    # Evaluate the model on the test set
    test_dataset = get_test_set(data_dir, add_noise)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    accuracy = evaluate_model(model, test_loader, device)
    if augmentation_name is not None:
        print(augmentation_name, ": ", end="")
    print(accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--add_noise', default=False, action="store_true",
                        help='Add noise to test data')

    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)