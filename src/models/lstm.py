#!/usr/bin/env python
"""
Defines an LSTM classifier for sequence-based coverage prediction.
"""

from torch import Tensor, nn


class LSTMClassifier(nn.Module):
	"""
	LSTM-based model that pools the final timestep for classification.

	Inputs:
	- input_size/hidden_size/num_layers/dropout/bidir/num_classes: Network shape and output size.

	Outputs:
	- forward returns logits for the requested number of classes.
	"""

	def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, bidir: bool, num_classes: int) -> None:
		"""
		Builds the LSTM backbone and classification head.

		Inputs:
		- input_size: Number of features per timestep.
		- hidden_size: Hidden dimension of the LSTM.
		- num_layers: Number of stacked LSTM layers.
		- dropout: Dropout rate between LSTM layers.
		- bidir: Whether to use a bidirectional LSTM.
		- num_classes: Number of output classes.

		Outputs:
		- Initializes the network modules.
		"""

		super().__init__()
		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
			bidirectional=bidir,
		)
		out_dim = hidden_size * (2 if bidir else 1)
		self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, num_classes))

	def forward(self, x: Tensor) -> Tensor:
		"""
		Runs a forward pass on a batch of sequences.

		Inputs:
		- x: Tensor shaped [batch, timesteps, features].

		Outputs:
		- logits: Tensor shaped [batch, num_classes].
		"""

		out, _ = self.lstm(x)
		last = out[:, -1, :]
		logits = self.head(last)
		return logits
