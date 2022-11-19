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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.
        """

        self.in_features = in_features
        self.out_features = out_features
        self.input_layer = input_layer
        self.params = {}

        if self.input_layer:
            self.params["weight"] = np.random.normal(size=(out_features, in_features), loc=0, scale=1 / np.sqrt(in_features))
        else:
            self.params["weight"] = np.random.normal(size=(out_features, in_features), loc=0, scale=np.sqrt(2) / np.sqrt(in_features))

        self.params["bias"] = np.zeros(shape=(1, self.out_features))

        self.grads = {}
        self.grads["weight"] = np.zeros(shape=(out_features, in_features))
        self.grads["bias"] = np.zeros(shape=(1, self.out_features))

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        self.layer_input = x.copy()
        out = self.layer_input @ self.params["weight"].T + self.params["bias"]

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        dx = dout @ self.params["weight"]
        self.grads["weight"] = dout.T @ self.layer_input
        self.grads["bias"] = np.ones(shape=(1, dout.shape[0])) @ dout

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        self.layer_input = None

        for grad in self.grads.keys():
            self.grads[grad] = np.zeros(shape=self.grads[grad].shape)

class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        """
        self.layer_input = x
        alpha = 1
        out = np.empty(shape=self.layer_input.shape)
        for i, batch in enumerate(self.layer_input):
            for j, sample in enumerate(batch):
                out[i, j] = sample if sample > 0 else alpha * (np.exp(sample) - 1)
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        """
        alpha = 1
        dx = np.empty(shape=self.layer_input.shape)
        for i, batch in enumerate(self.layer_input):
            for j, sample in enumerate(batch):
                dx[i, j] = 1 if sample > 0 else alpha * np.exp(sample)
        dx = dout * dx
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        self.layer_input = None


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        normalized_x = np.exp(x - np.max(x, axis=1, keepdims=True))

        exp_sum = np.sum(normalized_x, axis=1, keepdims=True)
        out = normalized_x / exp_sum
        self.y = out.copy()

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        self.ones_matrix = np.ones(shape=(dout.shape[1], dout.shape[1]))
        dx = self.y * (dout - (dout * self.y) @ self.ones_matrix)

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        self.y = None


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
        """
        T = np.eye(x.shape[1])[y]

        L = T * np.log(x)
        out = -np.sum(L) / x.shape[0]

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
        """
        T = np.eye(x.shape[1])[y]
        dx = (T / x) / -x.shape[0]

        return dx