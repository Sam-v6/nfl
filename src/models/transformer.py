#!/usr/bin/env python
"""
Defines the transformer architecture used for man/zone classification and holds a fcn to create a model.
"""

import torch
import torch.nn as nn


class ManZoneTransformer(nn.Module):
	"""
	Transformer encoder that ingests per-player features and predicts coverage.

	Inputs:
	- feature_len/model_dim/num_heads/num_layers/dim_feedforward/dropout/output_dim: Model shape and output size.

	Outputs:
	- forward returns logits for each class.
	"""

	def __init__(
		self,
		feature_len: int = 5,
		model_dim: int = 64,
		num_heads: int = 2,
		num_layers: int = 4,
		dim_feedforward: int = 256,
		dropout: float = 0.1,
		output_dim: int = 2,
	) -> None:
		"""
		Builds the transformer layers and pooling head.

		Inputs:
		- feature_len: Number of input features per player.
		- model_dim: Embedding dimension.
		- num_heads: Attention heads per encoder layer.
		- num_layers: Number of encoder layers.
		- dim_feedforward: Hidden size of the feedforward sublayer.
		- dropout: Dropout rate across the model.
		- output_dim: Number of output classes.

		Outputs:
		- Initialized model ready for training.
		"""

		super().__init__()
		self.feature_norm_layer = nn.BatchNorm1d(feature_len)

		self.feature_embedding_layer = nn.Sequential(
			nn.Linear(feature_len, model_dim),
			nn.ReLU(),
			nn.LayerNorm(model_dim),
			nn.Dropout(dropout),
		)

		transformer_encoder_layer = nn.TransformerEncoderLayer(
			d_model=model_dim,
			nhead=num_heads,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
		)
		self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)

		self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)

		self.decoder = nn.Sequential(
			nn.Linear(model_dim, model_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(model_dim, model_dim // 4),
			nn.ReLU(),
			nn.LayerNorm(model_dim // 4),
			nn.Linear(model_dim // 4, output_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Runs a forward pass over player features.

		Inputs:
		- x: Tensor shaped [batch, num_players, feature_len].

		Outputs:
		- logits: Tensor shaped [batch, output_dim] representing class scores.
		"""

		# x shape: (batch_size, num_players, feature_len)
		x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
		x = self.feature_embedding_layer(x)
		x = self.transformer_encoder(x)
		x = self.player_pooling_layer(x.permute(0, 2, 1)).squeeze(-1)
		x = self.decoder(x)
		return x


def create_transformer_model(config: dict[str, int | float]) -> ManZoneTransformer:
	"""
	Instantiates a ManZoneTransformer from a config mapping.

	Inputs:
	- config: Dictionary containing model_dim, num_heads, num_layers, multiplier, and dropout.

	Outputs:
	- model: Configured ManZoneTransformer instance.
	"""

	model = ManZoneTransformer(
		feature_len=5,  # num of input features (x, y, v_x, v_y, defense)
		model_dim=int(config["model_dim"]),  # from ray tune or loaded
		num_heads=int(config["num_heads"]),  # from ray tune or loaded
		num_layers=int(config["num_layers"]),  # from ray tune or loaded
		dim_feedforward=int(config["model_dim"]) * int(config["multiplier"]),  # from ray tune or loaded
		dropout=float(config["dropout"]),  # from ray tune or loaded
		output_dim=2,  # man or zone classification
	)

	return model
