import json
import tempfile
from pathlib import Path

import pytest
from test_model_bundle import MODEL_CONFIG
from test_training import training_fixture


@pytest.mark.integration
def test_one_cpu_trial_really_trains_and_restores_best_bundle(tmp_path: Path) -> None:
	import ray
	import torch

	from model_bundle import load_bundle, predict_proba
	from train_transformer import run_tune

	manifest = training_fixture(tmp_path / "data")
	# AF_UNIX sockets have a 107-byte path limit on Linux.
	ray_temp = tempfile.TemporaryDirectory(prefix="nfl-ray-")
	ray.init(num_cpus=2, include_dashboard=False, _temp_dir=ray_temp.name, object_store_memory=100 * 1024 * 1024)
	try:
		results = run_tune({**MODEL_CONFIG, "device": "cpu", "epochs": 2, "batch_size": 8}, manifest, tmp_path / "tune")
		assert len(results) == 1
		best = results.get_best_result(metric="macro_f1", mode="max")
		assert best.error is None
		assert best.metrics["training_iteration"] == 2
		assert best.checkpoint is not None
		with best.checkpoint.as_directory() as checkpoint:
			model, metadata = load_bundle(Path(checkpoint) / "bundle", torch.device("cpu"))
			assert metadata["dataset_hash"] == "fixture"
			assert metadata["best_epoch"] in (0, 1)
			input_data = torch.load(manifest.parent / "tune.pt", weights_only=True)["features"][:, -1]
			with torch.inference_mode():
				probs = predict_proba(model, input_data)
			assert torch.isfinite(probs).all()
			original, _ = load_bundle(Path(best.path) / "training/bundle", torch.device("cpu"))
			with torch.inference_mode():
				torch.testing.assert_close(probs, predict_proba(original, input_data))
		assert json.loads((Path(best.path) / "training/terminal.json").read_text())["status"] == "complete"
	finally:
		ray.shutdown()
		ray_temp.cleanup()
