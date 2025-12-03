#!/usr/bin/env python
"""
Wraps a LightGBM classifier for reuse.
"""

import lightgbm as lgb


class LGBModel:
	"""
	Thin wrapper around lightgbm.LGBMClassifier.

	Inputs:
	- n_estimators/class_weight/random_state: Core model settings.

	Outputs:
	- model attribute holds the configured classifier.
	"""

	def __init__(self, n_estimators: int = 100, class_weight: str = "balanced", random_state: int = 42) -> None:
		"""
		Builds the underlying LightGBM classifier.

		Inputs:
		- n_estimators: Number of boosting rounds.
		- class_weight: Strategy for class weighting.
		- random_state: Seed for reproducibility.

		Outputs:
		- Initializes the self.model attribute.
		"""

		self.model = lgb.LGBMClassifier(n_estimators=n_estimators, class_weight=class_weight, random_state=random_state)
