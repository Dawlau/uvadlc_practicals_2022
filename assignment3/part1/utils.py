################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np
import torch.nn as nn


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"

    eps = torch.randn(mean.shape).to(mean.device)
    z = mean + eps * std

    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """
    KLD = torch.exp(2 * log_std) + torch.pow(mean, 2) - 1 - 2 * log_std
    KLD /= 2
    KLD = torch.sum(KLD, axis=-1)

    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    batch, channels, heigth, width = img_shape
    dim_product = channels * heigth * width
    bpd = elbo * np.log2(np.e) / dim_product
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """
    percentiles = torch.Tensor([(i - 0.5) / grid_size for i in range(1, grid_size + 1)])
    normal = torch.distributions.Normal(0,1)

    p1, p2 = torch.meshgrid(percentiles, percentiles, indexing="xy")

    p1 = normal.icdf(p1)
    p2 = normal.icdf(p2)

    z = torch.stack([p1, p2], dim=-1)
    z = torch.flatten(z, end_dim=-2)

    logits = decoder(z)

    probabilities = nn.functional.softmax(logits, dim=1)
    probabilities = torch.permute(probabilities, (0, 2, 3, 1))
    probabilities = torch.flatten(probabilities, end_dim=2)

    x_samples = torch.multinomial(probabilities, 1).reshape(-1, 28, 28, 1)
    x_samples = torch.permute(x_samples, (0, 3, 1, 2))

    img_grid = make_grid(x_samples, nrow=grid_size).float()

    return img_grid

