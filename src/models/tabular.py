"""Inference adapter for a portable, hash-verified geometric baseline."""

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from torch import nn

from models.baselines import predict_baseline_proba


class TabularCoverageModel(nn.Module):
	def __init__(self, estimator: Pipeline | LGBMClassifier) -> None:
		super().__init__()
		self.estimator = estimator

	def forward(self, features: torch.Tensor) -> torch.Tensor:
		probabilities = predict_baseline_proba(self.estimator, features.detach().cpu().numpy())
		return torch.from_numpy(np.log(probabilities.clip(1e-15, 1)).astype(np.float32)).to(features.device)
