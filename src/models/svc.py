#!/usr/bin/env python
"""
Wraps a support vector classifier configured for balanced classes.
"""

from sklearn.svm import SVC


class SVCModel:
	"""
	Thin wrapper around sklearn's SVC with probability outputs.

	Inputs:
	- kernel/class_weight/random_state: Core model settings.

	Outputs:
	- model attribute holds the configured classifier.
	"""

	def __init__(self, kernel: str = "rbf", class_weight: str = "balanced", random_state: int = 42) -> None:
		"""
		Builds the underlying SVC model with probability estimates enabled.

		Inputs:
		- kernel: Kernel type for SVC.
		- class_weight: Strategy for class weighting.
		- random_state: Seed for reproducibility.

		Outputs:
		- Initializes the self.model attribute.
		"""
		self.model = SVC(kernel=kernel, class_weight=class_weight, probability=True, random_state=random_state)
