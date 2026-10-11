"""Probability captions for play views; labels describe predictions, not explanations."""

import numpy as np
import pandas as pd

from coverage_labels import CLASSES
from evaluate import PROB_COLUMNS


def coverage_caption(row: pd.Series) -> str:
	probabilities = np.asarray(row[PROB_COLUMNS], dtype=float)
	if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1, atol=1e-6):
		return "Coverage prediction unavailable"
	order = np.argsort(-probabilities, kind="stable")[:2]
	return "Predicted: " + " / ".join(f"{CLASSES[i]} {probabilities[i]:.0%}" for i in order)
