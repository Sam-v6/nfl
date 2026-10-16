"""Causal motion summaries for the bounded information diagnostic."""

import torch


def motion_summary(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	"""Sorted player displacements and nearest-opponent separation changes.

	Player slots must retain identity across time, as prepare_prefix guarantees.
	The caller restricts comparisons to the same plays with sufficient history.
	"""
	if mask.shape != x.shape[:2] or not mask[:, -1].all() or (mask.sum(1) < 2).any():
		raise ValueError("Motion requires at least two observed frames and an observed cutoff")
	first = x[torch.arange(len(x), device=x.device), mask.long().argmax(1)]
	last = x[:, -1]
	displacement = last[:, :, :2] - first[:, :, :2]
	defense = last[:, :, 4].eq(1)
	if not defense.sum(1).eq(11).all():
		raise ValueError("Motion requires eleven players on each side")
	parts = []
	for side in (defense, ~defense):
		for axis in (0, 1):
			parts.append(displacement[:, :, axis][side].reshape(len(x), 11).sort(1).values / 120)
	separations = []
	for frame in (first, last):
		d = frame[defense].reshape(len(x), 11, 5)
		o = frame[~defense].reshape(len(x), 11, 5)
		separations.append(torch.cdist(d[:, :, :2], o[:, :, :2]).amin(2))
	parts.append((separations[1] - separations[0]).sort(1).values / 120)
	return torch.cat(parts, dim=1)
