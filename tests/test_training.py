import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_model_bundle import MODEL_CONFIG


def training_fixture(root: Path) -> Path:
	from coverage_labels import CLASSES
	from dataset import FEATURES, file_hash

	root.mkdir()
	index = []
	shards = []
	for split, game_id, n in [("fit", 1, 32), ("tune", 2, 16)]:
		y = torch.arange(n) % 2
		x = torch.zeros(n, 1, 22, 5)
		x[:, :, :11, 4] = 1
		x[:, :, :, 0] = (y.float() * 2 - 1)[:, None, None] * 3
		mask = torch.ones(n, 1, dtype=torch.bool)
		name = f"{split}.pt"
		torch.save({"features": x, "time_mask": mask, "targets": y}, root / name)
		shards.append({"path": name, "rows": n, "sha256": file_hash(root / name)})
		index.extend(
			[
				{
					"season": 2022,
					"gameId": game_id,
					"playId": i,
					"week": 1 if split == "fit" else 5,
					"cutoff_frame": 99,
					"offset": -1,
					"split": split,
					"target": int(y[i]),
					"augmented": False,
					"shard": name,
					"shard_row": i,
				}
				for i in range(n)
			]
		)
	pd.DataFrame(index).to_parquet(root / "examples.parquet", index=False)
	manifest = {
		"status": "complete",
		"classes": list(CLASSES),
		"features": list(FEATURES),
		"settings": {"history_frames": 1},
		"dataset_hash": "fixture",
		"examples_sha256": file_hash(root / "examples.parquet"),
		"shards": shards,
	}
	path = root / "manifest.json"
	path.write_text(json.dumps(manifest))
	return path


def test_cpu_training_overfits_tiny_separable_dataset(tmp_path: Path) -> None:
	from model_bundle import load_bundle, predict_proba

	manifest = training_fixture(tmp_path / "data")
	module = importlib.import_module("training")
	# The production head has 16 channels; a two-channel LayerNorm head
	# collapses this fixture into near-constant signs and is not representative.
	bundle = module.train_model({**MODEL_CONFIG, "model_dim": 16, "device": "cpu", "epochs": 40, "batch_size": 32, "lr": 0.01, "weight_decay": 0.0, "seed": 42}, manifest, tmp_path / "run")
	model, _ = load_bundle(bundle, torch.device("cpu"))
	shard = torch.load(manifest.parent / "tune.pt", weights_only=True)
	with torch.inference_mode():
		predicted = predict_proba(model, shard["features"][:, -1]).argmax(dim=1)
	assert (predicted == shard["targets"]).float().mean() >= 0.95
	receipt = json.loads((bundle.parent / "terminal.json").read_text())
	assert receipt["status"] == "complete"
	assert receipt["device"] == "cpu"


def test_resumed_training_matches_uninterrupted_cpu_run(tmp_path: Path) -> None:
	module = importlib.import_module("training")
	manifest = training_fixture(tmp_path / "data")
	config = {**MODEL_CONFIG, "device": "cpu", "epochs": 4, "batch_size": 8, "lr": 0.01, "seed": 17}
	module.train_model(config, manifest, tmp_path / "whole")
	module.train_model({**config, "epochs": 2}, manifest, tmp_path / "resumed")
	module.train_model({**config, "resume": True}, manifest, tmp_path / "resumed")
	whole = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
	resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
	assert whole["epoch"] == resumed["epoch"] == 3
	for name, tensor in whole["model"].items():
		torch.testing.assert_close(tensor, resumed["model"][name], atol=0, rtol=0)
	assert whole["history"] == resumed["history"]


def test_latest_checkpoint_cannot_replace_better_bundle(tmp_path: Path) -> None:
	module = importlib.import_module("training")
	manifest = training_fixture(tmp_path / "data")
	module.train_model({**MODEL_CONFIG, "device": "cpu", "epochs": 5, "batch_size": 32, "lr": 0.05}, manifest, tmp_path / "run")
	latest = torch.load(tmp_path / "run/latest.pt", weights_only=True)
	metadata = json.loads((tmp_path / "run/bundle/metadata.json").read_text())
	best = max(latest["history"], key=lambda row: (row["macro_f1"], -row["log_loss"]))
	assert metadata["best_epoch"] == best["epoch"]


def test_interrupt_records_failure_and_can_resume_exactly(tmp_path: Path) -> None:
	import pytest

	from training import train_model

	manifest = training_fixture(tmp_path / "data")
	config = {**MODEL_CONFIG, "device": "cpu", "epochs": 4, "batch_size": 8, "seed": 73}

	def interrupt(metrics: dict, directory: Path) -> None:
		if metrics["epoch"] == 1:
			raise KeyboardInterrupt("injected interruption")

	with pytest.raises(KeyboardInterrupt):
		train_model(config, manifest, tmp_path / "interrupted", reporter=interrupt)
	assert json.loads((tmp_path / "interrupted/terminal.json").read_text())["status"] == "failed"
	train_model({**config, "resume": True}, manifest, tmp_path / "interrupted")
	train_model(config, manifest, tmp_path / "whole")
	a = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
	b = torch.load(tmp_path / "interrupted/latest.pt", weights_only=True)
	for name, tensor in a["model"].items():
		torch.testing.assert_close(tensor, b["model"][name], atol=0, rtol=0)
	assert a["history"] == b["history"]


def test_direct_and_loader_use_identical_shuffles(tmp_path: Path) -> None:
	from training import train_model

	manifest = training_fixture(tmp_path / "data")
	config = {**MODEL_CONFIG, "device": "cpu", "epochs": 3, "batch_size": 7, "seed": 42, "dropout": 0.1}
	train_model({**config, "input_pipeline": "loader"}, manifest, tmp_path / "loader")
	train_model({**config, "input_pipeline": "direct"}, manifest, tmp_path / "direct")
	a = torch.load(tmp_path / "loader/latest.pt", weights_only=True)
	b = torch.load(tmp_path / "direct/latest.pt", weights_only=True)
	for name, tensor in a["model"].items():
		torch.testing.assert_close(tensor, b["model"][name], atol=0, rtol=0)
	assert a["history"] == b["history"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_resume_honors_checkpoint_early_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool) -> None:
	import training

	manifest = training_fixture(tmp_path / "data")
	config = {**MODEL_CONFIG, "device": "cpu", "epochs": 10, "batch_size": 8, "patience": 1}
	monkeypatch.setattr(training, "_score", lambda *args: {"macro_f1": 0.5, "accuracy": 0.5, "log_loss": 1.0})
	training.train_model(config, manifest, tmp_path / "whole")

	def interrupt(metrics: dict, directory: Path) -> None:
		if metrics["epoch"] == 1:
			raise KeyboardInterrupt("after checkpoint, before early stop")

	if interrupted:
		with pytest.raises(KeyboardInterrupt):
			training.train_model(config, manifest, tmp_path / "resumed", reporter=interrupt)
	else:
		training.train_model(config, manifest, tmp_path / "resumed")
	training.train_model({**config, "resume": True}, manifest, tmp_path / "resumed")
	whole = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
	resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
	assert whole["epoch"] == resumed["epoch"] == 1
	assert whole["history"] == resumed["history"]
	for name, tensor in whole["model"].items():
		torch.testing.assert_close(tensor, resumed["model"][name], atol=0, rtol=0)
	assert json.loads((tmp_path / "resumed/terminal.json").read_text())["epochs_completed"] == 2
