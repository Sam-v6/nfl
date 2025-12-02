#!/usr/bin/env python
"""
Provides a simple feedforward neural network wrapper.
"""

import torch


class MLPModel(torch.nn.Module):
	"""
	Basic multilayer perceptron with one hidden layer.

	Inputs:
	- input_size/hidden_size/output_size: Layer dimensions.

	Outputs:
	- forward returns logits for the configured output size.
	"""

	def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
		"""
		Constructs the MLP layers.

		Inputs:
		- input_size: Number of input features.
		- hidden_size: Size of hidden layer.
		- output_size: Number of output units.

		Outputs:
		- Initializes the torch module layers.
		"""
		super().__init__()

		self.layers = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, output_size))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Runs a forward pass through the MLP.

		Inputs:
		- x: Input tensor of shape [batch, input_size].

		Outputs:
		- logits: Tensor of shape [batch, output_size].
		"""
		return self.layers(x)
