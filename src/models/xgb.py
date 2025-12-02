#!/usr/bin/env python
"""
Wraps a default XGBoost classifier for reuse.
"""

import xgboost as xgb


class XGBModel:
	"""
	Thin wrapper around xgboost.XGBClassifier.

	Inputs:
	- objective/n_estimators/random_state: Core model settings.

	Outputs:
	- model attribute holds the configured classifier.
	"""

	def __init__(self, objective: str = "binary:logistic", n_estimators: int = 100, random_state: int = 42) -> None:
		"""
		Builds the underlying XGBoost classifier.

		Inputs:
		- objective: Loss function to optimize.
		- n_estimators: Number of trees.
		- random_state: Seed for reproducibility.

		Outputs:
		- Initializes the self.model attribute.
		"""
		self.model = xgb.XGBClassifier(objective=objective, n_estimators=n_estimators, random_state=random_state)
