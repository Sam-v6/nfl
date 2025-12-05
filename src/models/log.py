#!/usr/bin/env python
"""
Wraps a default logistic regression classifier for reuse.
"""

from sklearn.linear_model import LogisticRegression


class LogisticModel:
	"""
	Thin wrapper around sklearn's LogisticRegression.

	Inputs:
	- max_iter/class_weight: Core model settings.

	Outputs:
	- model attribute holds the configured classifier.
	"""

	def __init__(self, max_iter: int = 1000, class_weight: str = "balanced") -> None:
		"""
		Builds the underlying LogisticRegression model.

		Inputs:
		- max_iter: Maximum solver iterations.
		- class_weight: Strategy for class weighting.

		Outputs:
		- Initializes the self.model attribute.
		"""

		self.model = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
