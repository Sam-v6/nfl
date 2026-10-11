import importlib
import json
from pathlib import Path

import pytest
import torch

MODEL_CONFIG = {"model_dim": 8, "num_heads": 2, "num_layers": 1, "multiplier": 2, "dropout": 0.0, "num_classes": 7, "feature_len": 5}


def test_seven_class_model_is_player_permutation_invariant() -> None:
	from models.transformer import create_transformer_model

	model = create_transformer_model(MODEL_CONFIG).eval()
	x = torch.randn(3, 22, 5)
	with torch.inference_mode():
		first = model(x)
		other = model(x[:, torch.randperm(22)])
	assert first.shape == (3, 7)
	torch.testing.assert_close(first, other, atol=1e-5, rtol=1e-4)


def test_portable_bundle_round_trip_and_class_order_validation(tmp_path: Path) -> None:
	from models.transformer import create_transformer_model

	module = importlib.import_module("model_bundle")
	model = create_transformer_model(MODEL_CONFIG).eval()
	path = tmp_path / "bundle"
	module.save_bundle(model, path, MODEL_CONFIG, "fixture")
	restored, metadata = module.load_bundle(path, torch.device("cpu"))
	x = torch.randn(2, 22, 5)
	with torch.inference_mode():
		torch.testing.assert_close(module.predict_proba(model, x), module.predict_proba(restored, x))
	assert len(metadata["classes"]) == 7
	info = json.loads((path / "metadata.json").read_text())
	info["classes"][0], info["classes"][1] = info["classes"][1], info["classes"][0]
	(path / "metadata.json").write_text(json.dumps(info))
	with pytest.raises(ValueError, match="vocabulary"):
		module.load_bundle(path, torch.device("cpu"))


def test_corrupt_weights_are_rejected(tmp_path: Path) -> None:
	from models.transformer import create_transformer_model

	module = importlib.import_module("model_bundle")
	module.save_bundle(create_transformer_model(MODEL_CONFIG), tmp_path, MODEL_CONFIG, "fixture")
	(tmp_path / "weights.pt").write_bytes(b"bad")
	with pytest.raises(ValueError, match="hash"):
		module.load_bundle(tmp_path, torch.device("cpu"))


def test_legacy_binary_requires_explicit_metadata(tmp_path: Path) -> None:
	from model_bundle import import_legacy_binary, load_bundle
	from models.transformer import create_transformer_model

	config = {**MODEL_CONFIG, "num_classes": 2}
	model = create_transformer_model(config).eval()
	weights = tmp_path / "legacy.pt"
	torch.save({"_orig_mod." + key: value for key, value in model.state_dict().items()}, weights)
	with pytest.raises(ValueError, match="explicit"):
		import_legacy_binary(weights, MODEL_CONFIG, tmp_path / "bad")
	bundle = import_legacy_binary(weights, config, tmp_path / "converted")
	loaded, metadata = load_bundle(bundle, torch.device("cpu"))
	assert metadata["classes"] == ["Zone", "Man"]
	x = torch.randn(3, 22, 5)
	with torch.inference_mode():
		torch.testing.assert_close(model(x), loaded(x), atol=0, rtol=0)


def test_tabular_baseline_is_portable_and_uses_same_class_order(tmp_path: Path) -> None:
	import numpy as np

	from coverage_labels import CLASSES
	from model_bundle import load_bundle, save_baseline_bundle
	from models.baselines import fit_baseline, predict_baseline_proba

	x = np.zeros((20, 22, 5), dtype=np.float32)
	x[:, :11, 4] = 1
	x[10:, :11, 0] = 10
	y = np.array([3] * 10 + [5] * 10)
	estimator = fit_baseline(x, y)
	bundle = save_baseline_bundle(estimator, tmp_path / "baseline", "fixture")
	model, metadata = load_bundle(bundle, torch.device("cpu"))
	assert metadata["classes"] == list(CLASSES)
	with torch.inference_mode():
		np.testing.assert_allclose(model(torch.from_numpy(x)).softmax(-1).numpy(), predict_baseline_proba(estimator, x), atol=1e-6)
