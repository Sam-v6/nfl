import importlib

import torch
from test_model_bundle import MODEL_CONFIG


def test_temporal_padding_and_player_permutation_invariance() -> None:
	module = importlib.import_module("models.temporal")
	model = module.TemporalCoverageModel({**MODEL_CONFIG, "model_dim": 16}).eval()
	x = torch.randn(2, 3, 22, 5)
	mask = torch.tensor([[False, True, True], [True, False, True]])
	with torch.inference_mode():
		expected = model(x, mask)
		changed = x.clone()
		changed[~mask] = 9999
		torch.testing.assert_close(expected, model(changed, mask), atol=1e-5, rtol=1e-4)
		permuted = x[:, :, torch.randperm(22)]
		torch.testing.assert_close(expected, model(permuted, mask), atol=1e-5, rtol=1e-4)
		x_padded = torch.cat([torch.full((2, 5, 22, 5), 9999.0), x], dim=1)
		mask_padded = torch.cat([torch.zeros(2, 5, dtype=torch.bool), mask], dim=1)
		torch.testing.assert_close(expected, model(x_padded, mask_padded), atol=1e-5, rtol=1e-4)
	assert expected.shape == (2, 7)


def test_short_temporal_history_has_finite_gradients() -> None:
	module = importlib.import_module("models.temporal")
	model = module.TemporalCoverageModel({**MODEL_CONFIG, "model_dim": 16})
	x = torch.randn(2, 3, 22, 5, requires_grad=True)
	mask = torch.tensor([[False, False, True], [True, False, True]])
	model(x, mask).sum().backward()
	assert x.grad is not None
	assert torch.isfinite(x.grad).all()
	assert not x.grad[~mask].any()
