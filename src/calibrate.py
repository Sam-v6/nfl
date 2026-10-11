"""Scalar calibration on the designated Week 6 role, with explicit provenance."""

import json
import math
import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from dataset import file_hash
from dataset_io import load_examples
from generate_predictions import predict_logits
from model_bundle import load_bundle
from training import select_device


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
	if not math.isfinite(temperature) or temperature <= 0:
		raise ValueError("Require a finite positive temperature")
	values = np.asarray(logits, dtype=np.float64)
	if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
		raise ValueError("Invalid calibration logits")
	values = values / temperature
	return np.exp(values - logsumexp(values, axis=1, keepdims=True))


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
	values = np.asarray(logits, dtype=np.float64)
	y = np.asarray(targets)
	apply_temperature(values, 1.0)
	if y.shape != (len(values),) or not np.isin(y, np.arange(values.shape[1])).all():
		raise ValueError("Invalid calibration targets")
	y = y.astype(int)

	def objective(log_temperature: float) -> float:
		scaled = values / math.exp(log_temperature)
		return float((np.asarray(logsumexp(scaled, axis=1)) - scaled[np.arange(len(y)), y]).mean())

	result = minimize_scalar(objective, bounds=(-3, 3), method="bounded", options={"xatol": 1e-8})
	if not result.success or not math.isfinite(result.fun):
		raise ValueError("Temperature fit failed")
	return math.exp(float(result.x))


def calibrate_bundle(bundle: Path, manifest_path: Path, output_dir: Path, split: str = "calibrate", device: str = "cpu") -> Path:
	if split != "calibrate":
		raise ValueError("Temperature fitting requires the calibrate role only")
	if output_dir.exists():
		raise ValueError("Calibrated bundle output already exists")
	selected_device = select_device(device)
	model, metadata = load_bundle(bundle, selected_device)
	x, mask, y, index = load_examples(manifest_path, "calibrate", offset=-1)
	if not index.season.eq(2022).all() or not index.week.eq(6).all():
		raise ValueError("Calibration IDs must belong to 2022 Week 6")
	logits = predict_logits(model, x, mask, 512, selected_device).numpy()
	if len(metadata["classes"]) == 2:
		y = ((y == 0) | (y == 1) | (y == 5)).long()
	targets = y.numpy()
	temperature = fit_temperature(logits, targets)
	probs = apply_temperature(logits, temperature)
	confidence = probs.max(1)
	correct = probs.argmax(1) == targets
	curve = []
	for threshold in np.linspace(0, 1, 101):
		answered = confidence >= threshold
		curve.append({"threshold": float(threshold), "coverage": float(answered.mean()), "count": int(answered.sum()), "accuracy": float(correct[answered].mean()) if answered.any() else None})
	eligible = [row for row in curve if row["count"] >= 20 and row["accuracy"] is not None and row["accuracy"] >= 0.75]
	threshold = max(eligible, key=lambda row: row["coverage"])["threshold"] if eligible else 1.000001
	shutil.copytree(bundle, output_dir, ignore=shutil.ignore_patterns("stage.json"))
	metadata.update(
		temperature=temperature,
		confidence_threshold=threshold,
		calibration={
			"split": "calibrate",
			"season": 2022,
			"week": 6,
			"offset": -1,
			"examples": len(y),
			"manifest_sha256": file_hash(manifest_path),
			"source_weights_sha256": metadata["weights_sha256"],
		},
	)
	(output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
	receipt = {
		"status": "complete",
		"temperature": temperature,
		"curve": curve,
		"log_loss_before": float(-np.log(apply_temperature(logits, 1)[np.arange(len(y)), targets].clip(1e-15, 1)).mean()),
		"log_loss_after": float(-np.log(probs[np.arange(len(y)), targets].clip(1e-15, 1)).mean()),
		"metadata_sha256": file_hash(output_dir / "metadata.json"),
	}
	(output_dir / "calibration.json").write_text(json.dumps(receipt, indent=2) + "\n")
	return output_dir
