import numpy as np
import pytest
import torch


def test_geometry_is_permutation_invariant_and_finite() -> None:
	from football_features import geometry_features

	x = torch.randn(3, 22, 5)
	x[:, :11, 4] = 1
	x[:, 11:, 4] = 0
	torch.testing.assert_close(geometry_features(x), geometry_features(x[:, torch.randperm(22)]), atol=1e-5, rtol=1e-4)
	assert torch.isfinite(geometry_features(x)).all()


def test_context_rejects_future_metadata() -> None:
	from test_dataset import frames

	from dataset import prepare_prefix
	from football_features import CONTEXT_FEATURES, build_context_features

	x, mask = prepare_prefix(frames(), 100, "DEF")
	import pandas as pd

	from dataset import FEATURES

	prefix = pd.DataFrame(x[mask].reshape(-1, 5), columns=list(FEATURES))
	a = build_context_features(prefix, {})
	assert a.shape == (len(CONTEXT_FEATURES),)
	assert np.isfinite(a).all()
	with pytest.raises(ValueError, match="allowlist"):
		build_context_features(prefix, {"targetReceiver": 123})
