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

"""Defines various kinds of visual-prompting modules for images."""
import torch
import torch.nn as nn
import numpy as np


class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        self.device = args.device

        pad_left_tensor  = torch.randn(1, 3, image_size - 2 * pad_size, pad_size)
        pad_right_tensor = torch.randn(1, 3, image_size - 2 * pad_size, pad_size)
        pad_up_tensor    = torch.randn(1, 3, pad_size, image_size)
        pad_down_tensor  = torch.randn(1, 3, pad_size, image_size)

        self.pad_left = torch.nn.Parameter(pad_left_tensor)
        self.pad_right = torch.nn.Parameter(pad_right_tensor)
        self.pad_up = torch.nn.Parameter(pad_up_tensor)
        self.pad_down = torch.nn.Parameter(pad_down_tensor)


    def forward(self, x):
        prompt = torch.zeros(x.shape)
        prompt = prompt.to(self.device)
        x = x.to(self.device)

        pad_size = self.pad_up.shape[-2]
        image_size = x.shape[-1]

        prompt[ : , : , : pad_size, : ] = self.pad_up
        prompt[ :, : , pad_size : image_size - pad_size, : pad_size] = self.pad_left
        prompt[ : , : , image_size - pad_size : , : ] = self.pad_down
        prompt[ : , : , pad_size : image_size - pad_size, image_size - pad_size : ] = self.pad_right

        prompted_x = x + prompt

        return prompted_x


class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        self.device = args.device
        self.patch = torch.nn.Parameter(torch.randn(1, 3, args.prompt_size, args.prompt_size))


    def forward(self, x):
        patch_row = 0
        patch_col = 0
        patch_size = self.patch.shape[-1]

        prompt = torch.zeros(x.shape).to(self.device)
        prompt[ : , : , patch_row : patch_row + patch_size, patch_col : patch_col + patch_size] = self.patch

        x = x.to(self.device)
        prompted_x = x + prompt

        return prompted_x


class RandomPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a random patch in the image.
    For refernece, this prompt should look like Fig 2(b) in the PDF.
    """
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        self.device = args.device
        self.patch = torch.nn.Parameter(torch.randn(1, 3, args.prompt_size, args.prompt_size))


    def forward(self, x):
        patch_size = self.patch.shape[-1]
        image_size = x.shape[-1]

        patch_row = np.random.randint(0, image_size - patch_size + 1)
        patch_col = np.random.randint(0, image_size - patch_size + 1)

        prompt = torch.zeros(x.shape).to(self.device)
        prompt[ : , : , patch_row : patch_row + patch_size, patch_col : patch_col + patch_size] = self.patch

        x = x.to(self.device)
        prompted_x = x + prompt

        return prompted_x


class FullPadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(FullPadPrompter, self).__init__()
        image_size = args.image_size
        self.device = args.device
        self.prompt = torch.nn.Parameter(-5000 * torch.rand(1, 3, image_size, image_size) + 2500)


    def forward(self, x):
        prompt = self.prompt.to(self.device)
        x = x.to(self.device)

        prompted_x = x + prompt

        return prompted_x