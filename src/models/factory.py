"""Explicit model selection and a shared input signature."""

from typing import Any

import torch
from torch import nn

from models.transformer import create_transformer_model


def create_coverage_model(config: dict[str, Any]) -> nn.Module:
	kind = config.get("model_type", "spatial")
	if kind == "spatial":
		return create_transformer_model(config)
	if kind == "temporal":
		from models.temporal import TemporalCoverageModel

		return TemporalCoverageModel(config)
	raise ValueError(f"Unknown neural model type: {kind}")


def forward_model(model: nn.Module, features: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
	if getattr(model, "requires_time_mask", False):
		return model(features, time_mask)
	return model(features[:, -1])
