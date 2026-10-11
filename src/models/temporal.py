"""Observed-frame spatial encoding and a causal GRU that skips masked history."""

from typing import Any

import torch
from torch import nn

from football_features import CONTEXT_FEATURES, geometry_features
from models.transformer import create_transformer_model


class TemporalCoverageModel(nn.Module):
	requires_time_mask = True

	def __init__(self, config: dict[str, Any]) -> None:
		super().__init__()
		width = int(config["model_dim"])
		self.frame_encoder = create_transformer_model(config)
		self.frame_encoder.decoder = nn.Identity()
		self.recurrent = nn.GRUCell(width, width)
		self.context = bool(config.get("context_features", False))
		self.decoder = nn.Linear(width + (len(CONTEXT_FEATURES) if self.context else 0), int(config.get("num_classes", 7)))

	def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
		if x.ndim != 4 or time_mask.shape != x.shape[:2] or time_mask.dtype != torch.bool:
			raise ValueError("Temporal input requires [batch,time,22,features] and boolean mask")
		observed = self.frame_encoder(x[time_mask])
		embeddings = observed.new_zeros((*x.shape[:2], observed.shape[-1]))
		embeddings[time_mask] = observed
		hidden = observed.new_zeros((len(x), observed.shape[-1]))
		for frame in range(x.shape[1]):
			updated = self.recurrent(embeddings[:, frame], hidden)
			hidden = torch.where(time_mask[:, frame, None], updated, hidden)
		if self.context:
			hidden = torch.cat([hidden, geometry_features(x[:, -1])], dim=-1)
		return self.decoder(hidden)
