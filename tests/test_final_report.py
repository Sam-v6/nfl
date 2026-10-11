"""Exercise real CPU training/calibration/scoring with a frozen synthetic dataset."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_model_bundle import MODEL_CONFIG

from dataset import file_hash


def fixture(root: Path) -> tuple[Path, dict]:
	from coverage_labels import CLASSES
	from dataset import FEATURES
	from training import DEFAULTS

	root.mkdir()
	rows, shards = [], []
	for split, week, offsets in [("fit", 1, [-1]), ("tune", 5, [-1]), ("calibrate", 6, [-1]), ("final", 7, [-10, -1]), ("historical", 9, [-1])]:
		y = torch.arange(28 * len(offsets)) % 7
		x = torch.zeros(len(y), 1, 22, 5)
		x[:, :, :11, 4] = 1
		x[:, :, :, 0] = y[:, None, None].float() / 7
		x[:, :, :, 1] = torch.arange(22)[None, None] / 22
		name = f"{split}.pt"
		torch.save({"features": x, "time_mask": torch.ones(len(y), 1, dtype=torch.bool), "targets": y}, root / name)
		shards.append({"path": name, "rows": len(y), "sha256": file_hash(root / name)})
		for i, target in enumerate(y):
			offset = offsets[i // 28]
			rows.append(
				{
					"season": 2022,
					"week": week,
					"gameId": week * 10 + (i % 28) // 14,
					"playId": i % 28,
					"cutoff_frame": 100 + offset,
					"offset": offset,
					"target": int(target),
					"split": split,
					"augmented": False,
					"shard": name,
					"shard_row": i,
				}
			)
	pd.DataFrame(rows).to_parquet(root / "examples.parquet", index=False)
	manifest = root / "manifest.json"
	manifest.write_text(
		json.dumps(
			{
				"status": "complete",
				"fixture": True,
				"classes": list(CLASSES),
				"features": list(FEATURES),
				"settings": {"history_frames": 1},
				"dataset_hash": "a" * 64,
				"examples_sha256": file_hash(root / "examples.parquet"),
				"shards": shards,
			}
		)
	)
	candidate = {**DEFAULTS, **MODEL_CONFIG, "model_dim": 16, "device": "cpu", "precision": "fp32", "epochs": 2, "cpu_threads": 2}
	benchmark = root / "benchmark.json"
	benchmark.write_text(
		json.dumps(
			{
				"status": "complete",
				"torch": str(torch.__version__),
				"manifest_sha256": file_hash(manifest),
				"device": "cpu",
				"batch_size": 512,
				"cpu_threads": 2,
				"tf32": False,
				"model_config": candidate,
				"model_source_sha256": {
					name: file_hash(Path(__file__).parents[1] / "src" / name) for name in ("models/transformer.py", "models/factory.py", "models/temporal.py", "football_features.py")
				},
				"inference_modes": [{"status": "complete", "parity": True, "precision": "fp32", "compile": False}],
			}
		)
	)
	return manifest, {"seeds": [17, 42, 73], "candidate": candidate, "benchmark": str(benchmark), "initial_attempts": 0, "initial_gpu_seconds": 0}


def test_three_seed_flow_recovers_training_and_late_failure_without_repeating_final_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	import final_report
	from experiment import run_experiment
	from training import train_model

	manifest, config = fixture(tmp_path / "data")
	attempts = [0]

	def trainer(settings: dict, manifest: Path, directory: Path, timeout: float) -> Path:
		attempts[0] += 1

		def interruption(metrics: dict, path: Path) -> None:
			if metrics["epoch"] == 0:
				raise KeyboardInterrupt("injected training interruption")

		return train_model(settings, manifest, directory, reporter=interruption if attempts[0] == 1 else None)

	monkeypatch.setattr(final_report, "run_experiment", lambda path, out: run_experiment(path, out, trainer=trainer))
	output = tmp_path / "comparison"
	terminal = final_report.run_final_comparison(manifest, config, output)
	assert json.loads(terminal.read_text())["status"] == "failed"
	selection_hash = file_hash(manifest.parent / "selection.json")
	original = final_report.predict_dataset
	failed = [False]

	def predict(*args: object, **kwargs: object) -> Path:
		if "historical" in str(args[2]) and not failed[0]:
			failed[0] = True
			raise RuntimeError("injected late interruption")
		return original(*args, **kwargs)

	monkeypatch.setattr(final_report, "predict_dataset", predict)
	with pytest.raises(RuntimeError, match="late interruption"):
		final_report.run_final_comparison(manifest, {**config, "resume": True}, output)
	cached = output / "seed-17/candidate-final-predictions/predictions.parquet"
	cached_time, cached_hash = cached.stat().st_mtime_ns, file_hash(cached)
	(output / "terminal.json").write_text("{corrupt interrupted terminal")
	terminal = final_report.run_final_comparison(manifest, {**config, "resume": True}, output)
	report = json.loads(terminal.read_text())
	assert report["status"] == "complete" and len(report["seed_results"]) == 3
	assert not report["promoted"] and not (tmp_path / "accepted-model.json").exists()
	assert cached.stat().st_mtime_ns == cached_time and file_hash(cached) == cached_hash
	assert file_hash(manifest.parent / "selection.json") == selection_hash
	assert report["training_receipt"]["attempts"] == 7
	assert final_report.run_final_comparison(manifest, {**config, "resume": True}, output) == terminal
	with pytest.raises(ValueError, match="identity"):
		final_report.run_final_comparison(manifest, {**config, "resume": True, "initial_attempts": 1}, output)


def test_mismatched_benchmark_refuses_freeze(tmp_path: Path) -> None:
	from final_report import run_final_comparison

	manifest, config = fixture(tmp_path / "data")
	benchmark = Path(config["benchmark"])
	data = json.loads(benchmark.read_text())
	data["batch_size"] = 1
	benchmark.write_text(json.dumps(data))
	with pytest.raises(ValueError, match="Benchmark"):
		run_final_comparison(manifest, config, tmp_path / "comparison")
	assert not (manifest.parent / "selection.json").exists()
