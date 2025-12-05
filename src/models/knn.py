#!/usr/bin/env python
"""
Wraps a default k-nearest-neighbor classifier for reuse.
"""

from sklearn.neighbors import KNeighborsClassifier


class KNNModel:
	"""
	Thin wrapper that instantiates a sklearn KNN classifier.

	Inputs:
	- n_neighbors: Number of neighbors to consider.

	Outputs:
	- model attribute holds the configured classifier.
	"""

	def __init__(self, n_neighbors: int = 5) -> None:
		"""
		Builds the underlying KNeighborsClassifier.

		Inputs:
		- n_neighbors: Number of neighbors to use.

		Outputs:
		- Initializes the self.model attribute.
		"""

		self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
