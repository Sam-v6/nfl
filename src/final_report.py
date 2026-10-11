"""A once-only frozen comparison; final scores never feed the training controller."""

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from calibrate import calibrate_bundle
from dataset import file_hash
from dataset_io import load_examples
from evaluate import KEYS, PROB_COLUMNS, _macro_from_cm, check_promotion, compare_predictions, evaluate_predictions, promote_bundle
from experiment import RunLock, _write, run_experiment
from generate_predictions import predict_dataset
from model_bundle import load_bundle, save_baseline_bundle
from models.baselines import fit_baseline
from training import DEFAULTS


def _binary_metrics(predictions: pd.DataFrame, index: pd.DataFrame) -> dict[str, Any]:
	joined = predictions[KEYS + ["prob_0", "prob_1"]].merge(index[KEYS + ["target", "offset"]], on=KEYS, validate="one_to_one")
	y = np.isin(joined.target.to_numpy(), [0, 1, 5]).astype(int)
	p = joined[["prob_0", "prob_1"]].to_numpy()
	pred = p.argmax(1)
	return {
		"eligible_plays": len(y),
		"accuracy": float((pred == y).mean()),
		"macro_f1": _macro_from_cm(np.asarray(confusion_matrix(y, pred, labels=[0, 1]))),
		"log_loss": float(-np.log(p[np.arange(len(y)), y].clip(1e-15, 1)).mean()),
	}


def _cached_stage(directory: Path, identity: dict[str, Any], produce: Callable[[Path], Path]) -> Path:
	"""Commit completed stages atomically; verify and reuse them on frozen recovery."""
	partial = directory.with_name(directory.name + ".partial")
	for existing in (directory, partial):
		if (existing / "stage.json").exists():
			receipt = json.loads((existing / "stage.json").read_text())
			if receipt["identity"] != identity or any(file_hash(existing / name) != digest for name, digest in receipt["files"].items()):
				raise ValueError("Cached stage provenance mismatch")
			if existing == partial:
				partial.replace(directory)
			return directory / receipt["result"]
	if directory.exists():
		raise ValueError("Completed stage has no receipt")
	if partial.exists():
		shutil.rmtree(partial)
	result = produce(partial)
	receipt = {
		"identity": identity,
		"result": str(result.relative_to(partial)),
		"files": {str(p.relative_to(partial)): file_hash(p) for p in partial.rglob("*") if p.is_file() and p.name != "stage.json"},
	}
	_write(partial / "stage.json", receipt)
	partial.replace(directory)
	return directory / receipt["result"]


def _verify_benchmark(benchmark: dict[str, Any], candidate: dict[str, Any], manifest_path: Path) -> None:
	architecture = ("model_type", "model_dim", "num_heads", "num_layers", "multiplier", "dropout", "num_classes", "feature_len")
	if (
		benchmark.get("status") != "complete"
		or benchmark.get("manifest_sha256") != file_hash(manifest_path)
		or benchmark.get("torch") != str(torch.__version__)
		or benchmark.get("device") != candidate["device"]
		or benchmark.get("batch_size") != candidate["batch_size"]
		or benchmark.get("cpu_threads") != candidate["cpu_threads"]
		or benchmark.get("tf32") is not candidate["tf32"]
		or any(benchmark.get("model_config", {}).get(key, DEFAULTS.get(key)) != candidate[key] for key in architecture)
		or not any(
			m.get("status") == "complete" and m.get("parity") is True and m.get("precision") == "fp32" and m.get("compile") is candidate["compile"] for m in benchmark.get("inference_modes", [])
		)
	):
		raise ValueError("Benchmark does not match the selected manifest, architecture or runtime")
	for name in ("models/transformer.py", "models/factory.py", "models/temporal.py", "football_features.py"):
		if benchmark.get("model_source_sha256", {}).get(name) != file_hash(Path(__file__).parent / name):
			raise ValueError("Benchmark model source mismatch")


def run_final_comparison(manifest_path: Path, config: dict[str, Any], output_dir: Path) -> Path:
	if output_dir.exists() and not config.get("resume"):
		raise ValueError("Frozen comparison output already exists; resume explicitly")
	# One dataset owner prevents concurrent freeze attempts, including different outputs.
	with RunLock(manifest_path.parent / "selection.lock"):
		return _run_final_comparison(manifest_path, config, output_dir)


def _run_final_comparison(manifest_path: Path, config: dict[str, Any], output_dir: Path) -> Path:
	manifest = json.loads(manifest_path.read_text())
	if config.get("seeds") != [17, 42, 73]:
		raise ValueError("Frozen comparison requires seeds 17, 42, 73")
	candidate = {**DEFAULTS, **config["candidate"]}
	if candidate["num_classes"] != 7 or candidate["model_type"] != "spatial":
		raise ValueError("This selected comparison supports the seven-class spatial finalist")
	benchmark_path = Path(config["benchmark"])
	benchmark = json.loads(benchmark_path.read_text())
	_verify_benchmark(benchmark, candidate, manifest_path)
	source_root = Path(__file__).parent
	config_hash = hashlib.sha256(json.dumps({k: v for k, v in config.items() if k != "resume"}, sort_keys=True).encode()).hexdigest()
	source_hashes = {str(p.relative_to(source_root)): file_hash(p) for p in source_root.rglob("*.py")}
	selection_path = manifest_path.parent / "selection.json"
	if selection_path.exists():
		selection = json.loads(selection_path.read_text())
		if not config.get("resume") or selection.get("output_dir") != str(output_dir.resolve()) or selection.get("config_hash") != config_hash:
			raise ValueError("Dataset already has a frozen selection with a different identity")
		if (
			selection.get("lock_sha256") != file_hash(source_root.parent / "uv.lock")
			or selection["source_hashes"] != source_hashes
			or selection["manifest_sha256"] != file_hash(manifest_path)
			or selection["benchmark_sha256"] != file_hash(benchmark_path)
		):
			raise ValueError("Frozen source, manifest or benchmark changed")
		output_dir.mkdir(parents=True, exist_ok=True)
	else:
		if output_dir.exists():
			raise ValueError("Existing comparison has no frozen identity")
		selection = {
			"frozen": True,
			"dataset_hash": manifest["dataset_hash"],
			"manifest_sha256": file_hash(manifest_path),
			"candidate": candidate,
			"baseline": "LightGBM, 100 trees, depth 4, 15 leaves, all matched fit endpoints",
			"seeds": config["seeds"],
			"deployment_seed": 42,
			"promotion_rule": "all three seed-specific gates must pass",
			"code_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root, text=True, capture_output=True, check=True).stdout.strip(),
			"lock_sha256": file_hash(source_root.parent / "uv.lock"),
			"source_hashes": source_hashes,
			"config_hash": config_hash,
			"output_dir": str(output_dir.resolve()),
			"runtime": {"torch": str(torch.__version__), "inference_precision": "fp32", "batch_size": 512},
			"benchmark_sha256": file_hash(benchmark_path),
		}
		_write(selection_path, selection)
		output_dir.mkdir(parents=True)
	_write(output_dir / "selection.json", selection)
	terminal_path = output_dir / "terminal.json"
	if terminal_path.exists():
		try:
			if json.loads(terminal_path.read_text()).get("status") == "complete":
				return terminal_path
		except json.JSONDecodeError:
			# A pre-atomic-write interruption may leave a terminal fragment.
			# Completed stages still have their independently verified receipts.
			pass
	try:
		return _finish_comparison(manifest_path, manifest, config, candidate, selection_path, output_dir)
	except BaseException as exc:
		_write(terminal_path, {"status": "failed", "reason": str(exc) or type(exc).__name__, "selection_hash": file_hash(selection_path)})
		raise


def _finish_comparison(manifest_path: Path, manifest: dict[str, Any], config: dict[str, Any], candidate: dict[str, Any], selection_path: Path, output_dir: Path) -> Path:
	selection_hash = file_hash(selection_path)
	trials = []
	for seed in config["seeds"]:
		trials.append({**candidate, "seed": seed})
		trials.append({**candidate, "seed": seed, "num_classes": 2})
	training_config = {
		"resume": (output_dir / "training/ledger.json").exists(),
		"manifest": str(manifest_path.resolve()),
		"stage": "confirm",
		"max_trials": 24,
		"gpu_hours": 24,
		"reserve_hours": 12,
		"trial_seconds": 3600,
		"initial_gpu_seconds": float(config.get("initial_gpu_seconds", 3600)),
		"initial_attempts": int(config.get("initial_attempts", 9)),
		"trials": trials,
	}
	training_config_path = output_dir / "training-config.json"
	_write(training_config_path, training_config)
	terminal = run_experiment(training_config_path, output_dir / "training")
	if json.loads(terminal.read_text())["status"] != "complete":
		_write(output_dir / "terminal.json", {"status": "failed", "reason": "Confirmation training incomplete", "training_receipt": str(terminal)})
		return output_dir / "terminal.json"
	# Every transformation below was chosen before the final-role features are read.
	x_fit, _, y_fit, _ = load_examples(manifest_path, "fit")
	seed_results = []
	benchmark = json.loads(Path(config["benchmark"]).read_text())
	_verify_benchmark(benchmark, candidate, manifest_path)
	if file_hash(Path(config["benchmark"])) != json.loads(selection_path.read_text())["benchmark_sha256"]:
		raise ValueError("Frozen benchmark changed")
	benchmark_verified = True
	for position, seed in enumerate(config["seeds"]):
		seed_dir = output_dir / f"seed-{seed}"
		seed_dir.mkdir(exist_ok=True)
		base = _cached_stage(
			seed_dir / "baseline",
			{"selection": selection_hash, "seed": seed},
			lambda p, seed=seed: save_baseline_bundle(fit_baseline(x_fit[:, -1].numpy(), y_fit.numpy(), "lightgbm", seed), p, manifest["dataset_hash"], seed),
		)
		neural = output_dir / "training" / f"trial-{position * 2:03d}" / "bundle"
		binary = output_dir / "training" / f"trial-{position * 2 + 1:03d}" / "bundle"
		bundles = {
			name: _cached_stage(
				seed_dir / f"{name}-calibrated",
				{"selection": selection_hash, "source_metadata": file_hash(path / "metadata.json")},
				lambda p, path=path: calibrate_bundle(path, manifest_path, p, device="cpu"),
			)
			for name, path in [("candidate", neural), ("baseline", base), ("binary", binary)]
		}
		metrics, predictions = {}, {}
		for name in ("candidate", "baseline"):
			p = _cached_stage(
				seed_dir / f"{name}-final-predictions",
				{"selection": selection_hash, "metadata": file_hash(bundles[name] / "metadata.json")},
				lambda p, bundle=bundles[name]: predict_dataset(bundle, manifest_path, p, 512, split="final", device="cpu"),
			)
			predictions[name] = pd.read_parquet(p)
			metric_path = _cached_stage(
				seed_dir / f"{name}-final-report", {"selection": selection_hash, "predictions": file_hash(p)}, lambda out, frame=predictions[name]: evaluate_predictions(frame, manifest_path, out)
			)
			metrics[name] = json.loads(metric_path.read_text())
			# Week 9 is a separately labeled compatibility report, never selection input.
			p_history = _cached_stage(
				seed_dir / f"{name}-historical-predictions",
				{"selection": selection_hash, "metadata": file_hash(bundles[name] / "metadata.json")},
				lambda p, bundle=bundles[name]: predict_dataset(bundle, manifest_path, p, 512, split="historical", device="cpu"),
			)
			_cached_stage(
				seed_dir / f"{name}-historical-report",
				{"selection": selection_hash, "predictions": file_hash(p_history)},
				lambda out, history=p_history: evaluate_predictions(pd.read_parquet(history), manifest_path, out),
			)
		binary_path = _cached_stage(
			seed_dir / "binary-final-predictions",
			{"selection": selection_hash, "metadata": file_hash(bundles["binary"] / "metadata.json")},
			lambda p, bundle=bundles["binary"]: predict_dataset(bundle, manifest_path, p, 512, split="final", device="cpu", offset=-1),
		)
		index = pd.read_parquet(manifest_path.parent / "examples.parquet").query('split == "final" and not augmented')
		binary_metrics = _binary_metrics(pd.read_parquet(binary_path), index)
		frames = {name: frame.loc[frame.offset.eq(-1), KEYS + PROB_COLUMNS].merge(index[KEYS + ["target"]], on=KEYS, validate="one_to_one") for name, frame in predictions.items()}
		comparison = compare_predictions(frames["candidate"], frames["baseline"])
		_, metadata = load_bundle(bundles["candidate"], torch.device("cpu"))
		comparison.update(
			secondary_candidate_f1=metrics["candidate"]["final/-10"]["macro_f1"],
			secondary_baseline_f1=metrics["baseline"]["final/-10"]["macro_f1"],
			binary_reference_f1=binary_metrics["macro_f1"],
			evidence={
				"seeds": config["seeds"],
				"benchmark_verified": benchmark_verified,
				"calibration_split": "calibrate",
				"selection_hash": selection_hash,
				"weights_sha256": metadata["weights_sha256"],
				"metadata_sha256": file_hash(bundles["candidate"] / "metadata.json"),
				"manifest_sha256": file_hash(manifest_path),
				"dataset_hash": manifest["dataset_hash"],
			},
		)
		decision = check_promotion(metrics["candidate"]["final/-1"], metrics["baseline"]["final/-1"], comparison)
		result = {"seed": seed, "candidate": metrics["candidate"], "baseline": metrics["baseline"], "binary_reference": binary_metrics, "comparison": comparison, "decision": decision}
		(seed_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
		seed_results.append(result)
		if seed == 42:
			uncertain = frames["candidate"].copy()
			uncertain["confidence"] = uncertain[PROB_COLUMNS].max(axis=1)
			uncertain.sort_values("confidence").head(50).to_parquet(output_dir / "uncertainty-examples.parquet", index=False)
	approved = all(row["decision"]["approved"] for row in seed_results)
	chosen = next(row for row in seed_results if row["seed"] == 42)
	decision = {**chosen["decision"], "approved": approved, "seed_decisions": {str(row["seed"]): row["decision"]["gates"] for row in seed_results}}
	promoted = promote_bundle(output_dir / "seed-42/candidate-calibrated", decision, output_dir.parent / "accepted-model.json", fixture=bool(manifest.get("fixture")))
	report = {
		"status": "complete",
		"promoted": promoted,
		"selection_hash": selection_hash,
		"seed_results": seed_results,
		"deployment_seed": 42,
		"baseline_bundle": str((output_dir / "seed-42/baseline-calibrated").resolve()),
		"candidate_bundle": str((output_dir / "seed-42/candidate-calibrated").resolve()),
		"training_receipt": json.loads(terminal.read_text()),
		"benchmark": str(Path(config["benchmark"]).resolve()),
		"limitations": [
			"Weeks 7–8 were exposed during historical development; this is a held-out comparison under the new protocol, not fresh-data generalization.",
			"Unknown/nonstandard labels and incomplete tracking are excluded with the dataset receipt; no novel-class detection claim.",
		],
	}
	path = output_dir / "terminal.json"
	_write(path, report)
	return path
