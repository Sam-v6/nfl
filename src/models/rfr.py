#!/usr/bin/env python
"""
Wraps a default random forest classifier for reuse.
"""

from sklearn.ensemble import RandomForestClassifier


class RFRModel:
	"""
	Thin wrapper around sklearn's RandomForestClassifier.

	Inputs:
	- n_estimators/class_weight/random_state: Core model settings.

	Outputs:
	- model attribute holds the configured classifier.
	"""

	def __init__(self, n_estimators: int = 100, class_weight: str = "balanced", random_state: int = 42) -> None:
		"""
		Builds the underlying RandomForestClassifier.

		Inputs:
		- n_estimators: Number of trees.
		- class_weight: Strategy for class weighting.
		- random_state: Seed for reproducibility.

		Outputs:
		- Initializes the self.model attribute.
		"""
		self.model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, random_state=random_state)
