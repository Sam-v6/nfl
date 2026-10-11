"""Cutoff-available geometry; no eventual assignments or play outcomes."""

import numpy as np
import pandas as pd
import torch

from dataset import FEATURES

CONTEXT_FEATURES = ("defense_mean_depth", "defense_depth_std", "defense_width", "offense_width", "nearest_opponent_mean", "defense_mean_vx", "defense_mean_vy", "offense_mean_vx", "offense_mean_vy")


def geometry_features(x: torch.Tensor) -> torch.Tensor:
	"""Nine bounded geometry summaries, with role masking independent of player order."""
	defense = x[:, :, 4].eq(1)
	offense = ~defense

	def mean(values: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
		return (values * side).sum(-1) / side.sum(-1).clamp_min(1)

	off_x = mean(x[:, :, 0], offense)
	depth = x[:, :, 0] - off_x[:, None]
	def_depth = mean(depth, defense)
	variance = mean((depth - def_depth[:, None]).square(), defense)

	def width(side: torch.Tensor) -> torch.Tensor:
		return x[:, :, 1].masked_fill(~side, -1e4).amax(-1) - x[:, :, 1].masked_fill(~side, 1e4).amin(-1)

	distances = torch.cdist(x[:, :, :2], x[:, :, :2])
	opponents = defense[:, :, None] & offense[:, None, :]
	nearest = distances.masked_fill(~opponents, 1e4).amin(-1)
	return torch.stack(
		[
			def_depth / 120,
			variance.clamp_min(1e-8).sqrt() / 120,
			width(defense) / 53.3,
			width(offense) / 53.3,
			mean(nearest, defense) / 120,
			mean(x[:, :, 2], defense) / 15,
			mean(x[:, :, 3], defense) / 15,
			mean(x[:, :, 2], offense) / 15,
			mean(x[:, :, 3], offense) / 15,
		],
		dim=-1,
	)


def build_context_features(tracking_prefix: pd.DataFrame, context: dict[str, object]) -> np.ndarray:
	if context:
		raise ValueError("Context allowlist currently includes geometry only")
	rows = tracking_prefix.iloc[-22:]
	if len(rows) != 22 or not set(FEATURES).issubset(rows.columns) or rows.defense.sum() != 11:
		raise ValueError("Geometry requires a complete prepared cutoff frame")
	x = torch.from_numpy(rows[list(FEATURES)].to_numpy(dtype=np.float32))[None]
	return geometry_features(x)[0].numpy()
