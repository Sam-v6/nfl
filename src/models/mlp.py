#!/usr/bin/env python

"""
Contains class for default multi-layer-perceptron model for general usage
"""

import torch


class MLPModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()


        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)