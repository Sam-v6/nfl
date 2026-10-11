"""Play-level metrics and paired game-bootstrap comparisons."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from coverage_labels import CLASSES
from dataset import file_hash

KEYS = ["season", "gameId", "playId", "cutoff_frame"]
PROB_COLUMNS = [f"prob_{i}" for i in range(7)]


def score_predictions(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
	targets = np.asarray(targets)
	p = np.asarray(probabilities, dtype=np.float64)
	if not len(targets) or p.shape != (len(targets), 7) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
		raise ValueError("Invalid probabilities: require finite seven-class distributions summing to one")
	if not np.isin(targets, np.arange(7)).all():
		raise ValueError("Invalid target class")
	targets = targets.astype(int)
	predicted = p.argmax(axis=1)
	matrix = np.asarray(confusion_matrix(targets, predicted, labels=np.arange(7)))
	support = matrix.sum(axis=1)
	predicted_support = matrix.sum(axis=0)
	precision = np.divide(np.diag(matrix), predicted_support, out=np.zeros(7), where=predicted_support > 0)
	recall = np.divide(np.diag(matrix), support, out=np.zeros(7), where=support > 0)
	f1 = np.divide(2 * np.diag(matrix), support + predicted_support, out=np.zeros(7), where=support + predicted_support > 0)
	man_prob = p[:, [0, 1, 5]].sum(axis=1)
	man_true = np.isin(targets, [0, 1, 5]).astype(int)
	man_pred = (man_prob >= 0.5).astype(int)
	binary_loss = -np.log(np.where(man_true == 1, man_prob, 1 - man_prob).clip(1e-15, 1)).mean()
	confidence = p.max(axis=1)
	correct = predicted == targets
	bins = np.minimum((confidence * 10).astype(int), 9)
	calibration = []
	for i in range(10):
		selected = bins == i
		if selected.any():
			calibration.append({"count": int(selected.sum()), "confidence": float(confidence[selected].mean()), "accuracy": float(correct[selected].mean())})
	return {
		"eligible_plays": len(targets),
		"accuracy": float(correct.mean()),
		"macro_f1": float(f1.mean()),
		"top2_accuracy": float(np.any(np.argsort(-p, axis=1, kind="stable")[:, :2] == targets[:, None], axis=1).mean()),
		"log_loss": float(-np.log(p[np.arange(len(targets)), targets].clip(1e-15, 1)).mean()),
		"brier": float(((p - np.eye(7)[targets]) ** 2).sum(axis=1).mean()),
		"ece": sum(abs(b["confidence"] - b["accuracy"]) * b["count"] / len(targets) for b in calibration),
		"calibration": calibration,
		"confusion": confusion_matrix(targets, predicted, labels=np.arange(7)).tolist(),
		"per_class": {
			label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i]), "sufficient_support": bool(support[i] >= 20)}
			for i, label in enumerate(CLASSES)
		},
		"binary": {"accuracy": float((man_true == man_pred).mean()), "macro_f1": _macro_from_cm(np.asarray(confusion_matrix(man_true, man_pred, labels=[0, 1]))), "log_loss": float(binary_loss)},
	}


def _validate_keys(frame: pd.DataFrame) -> None:
	if np.any(pd.isna(frame[KEYS].to_numpy())):
		raise ValueError("Missing prediction keys")
	if frame.duplicated(KEYS).any():
		raise ValueError("Duplicate prediction keys")


def evaluate_predictions(predictions: pd.DataFrame, manifest_path: Path, output_dir: Path) -> Path:
	manifest = json.loads(manifest_path.read_text())
	if manifest["classes"] != list(CLASSES):
		raise ValueError("Class vocabulary mismatch")
	index_path = manifest_path.parent / "examples.parquet"
	if file_hash(index_path) != manifest["examples_sha256"]:
		raise ValueError("Dataset index hash mismatch")
	expected = pd.read_parquet(index_path).query("not augmented")
	_validate_keys(predictions)
	joined = predictions.drop(columns=["target"], errors="ignore").merge(expected, on=KEYS, how="left", validate="one_to_one", indicator=True)
	if not np.all(joined["_merge"].to_numpy() == "both"):
		raise ValueError("Prediction keys outside the dataset")
	if np.any(joined["split"].to_numpy() == "final"):
		selection = manifest_path.parent / "selection.json"
		if not selection.exists():
			raise ValueError("Final evaluation requires frozen selection receipt")
		receipt = json.loads(selection.read_text())
		if not receipt.get("frozen") or receipt.get("dataset_hash") != manifest["dataset_hash"]:
			raise ValueError("Invalid selection receipt")
	reports = {}
	for _, group in joined.groupby(["split", "offset"], sort=True):
		split, offset = str(group.split.iloc[0]), int(group.offset.iloc[0])
		total = len(expected.loc[expected.split.eq(split) & expected.offset.eq(offset)])
		if len(group) != total:
			raise ValueError(f"Missing predictions for {split}/{offset}: {len(group)} of {total}")
		report = score_predictions(group.target.to_numpy(), group[PROB_COLUMNS].to_numpy())
		answered = group["answered"].to_numpy(dtype=bool) if "answered" in group else np.ones(len(group), dtype=bool)
		report["answer_fraction"] = float(answered.mean())
		report["answered_accuracy"] = float((group[PROB_COLUMNS].to_numpy()[answered].argmax(axis=1) == group.target.to_numpy()[answered]).mean()) if answered.any() else None
		reports[f"{split}/{offset}"] = report
	output_dir.mkdir(parents=True, exist_ok=False)
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	for key, report in reports.items():
		name = key.replace("/", "_")
		fig, ax = plt.subplots(figsize=(8, 6))
		image = ax.imshow(report["confusion"], cmap="Blues")
		ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted coverage", ylabel="Actual coverage", title=key)
		plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
		fig.colorbar(image, ax=ax)
		fig.tight_layout()
		fig.savefig(output_dir / f"confusion_{name}.png")
		plt.close(fig)
		fig, ax = plt.subplots()
		ax.plot([0, 1], [0, 1], "--", color="gray")
		ax.plot([b["confidence"] for b in report["calibration"]], [b["accuracy"] for b in report["calibration"]], "o-")
		ax.set(xlabel="Predicted confidence", ylabel="Observed accuracy", title=key, xlim=(0, 1), ylim=(0, 1))
		fig.tight_layout()
		fig.savefig(output_dir / f"calibration_{name}.png")
		plt.close(fig)
	path = output_dir / "metrics.json"
	path.write_text(json.dumps(reports, indent=2) + "\n")
	(output_dir / "provenance.json").write_text(json.dumps({"dataset_hash": manifest["dataset_hash"], "example_count": len(joined)}, indent=2) + "\n")
	return path


def _macro_from_cm(matrix: np.ndarray) -> float:
	denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
	return float(np.divide(2 * np.diag(matrix), denominator, out=np.zeros(len(matrix)), where=denominator > 0).mean())


def compare_predictions(candidate: pd.DataFrame, baseline: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
	for frame in (candidate, baseline):
		_validate_keys(frame)
		score_predictions(frame.target.to_numpy(), frame[PROB_COLUMNS].to_numpy())
	left, right = candidate.sort_values(KEYS).reset_index(drop=True), baseline.sort_values(KEYS).reset_index(drop=True)
	if not left[KEYS].equals(right[KEYS]) or not left.target.equals(right.target):
		raise ValueError("Comparison requires identical example keys and targets")
	games = left.groupby(["season", "gameId"], sort=True).indices
	if len(games) < 2:
		raise ValueError("At least two games required for paired bootstrap")
	left_pred, right_pred = left[PROB_COLUMNS].to_numpy().argmax(axis=1), right[PROB_COLUMNS].to_numpy().argmax(axis=1)
	left_cm, right_cm = [], []
	for indices in games.values():
		targets = left.target.to_numpy()[indices]
		left_cm.append(confusion_matrix(targets, left_pred[indices], labels=range(7)))
		right_cm.append(confusion_matrix(targets, right_pred[indices], labels=range(7)))
	left_cms, right_cms = np.stack(left_cm), np.stack(right_cm)
	rng = np.random.default_rng(seed)
	deltas = []
	for _ in range(2000):
		sample = rng.integers(0, len(games), len(games))
		deltas.append(_macro_from_cm(left_cms[sample].sum(axis=0)) - _macro_from_cm(right_cms[sample].sum(axis=0)))
	return {
		"macro_f1_delta": _macro_from_cm(left_cms.sum(axis=0)) - _macro_from_cm(right_cms.sum(axis=0)),
		"ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
		"n_games": len(games),
		"replicates": 2000,
		"excluded_replicates": 0,
		"seed": seed,
	}
