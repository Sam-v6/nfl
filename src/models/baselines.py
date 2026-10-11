"""Small geometric comparators with a fixed seven-class prediction contract."""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def geometric_summary(features: np.ndarray) -> np.ndarray:
	x = np.asarray(features)
	if x.ndim != 3 or x.shape[1:] != (22, 5) or not np.isfinite(x).all():
		raise ValueError("Expected finite [batch,22,5] features")
	rows = []
	for frame in x:
		defense, offense = frame[frame[:, 4] == 1], frame[frame[:, 4] == 0]
		if len(defense) != 11 or len(offense) != 11:
			raise ValueError("Expected eleven players on each side")
		distances = np.linalg.norm(defense[:, None, :2] - offense[None, :, :2], axis=-1)
		rows.append(
			np.concatenate(
				[
					np.sort(defense[:, 0] - offense[:, 0].mean()),
					np.sort(defense[:, 1]),
					np.sort(distances.min(axis=1)),
					np.sort(offense[:, 1]),
					defense[:, 2:4].mean(axis=0),
					offense[:, 2:4].mean(axis=0),
				]
			)
		)
	return np.stack(rows)


def fit_baseline(features: np.ndarray, targets: np.ndarray, kind: str = "logistic", seed: int = 42) -> Pipeline | LGBMClassifier:
	if kind == "logistic":
		model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
	elif kind == "lightgbm":
		model = LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, n_jobs=2, random_state=seed, verbosity=-1)
	else:
		raise ValueError(f"Unknown baseline: {kind}")
	return model.fit(pd.DataFrame(geometric_summary(features)), targets)


def predict_baseline_proba(model: Pipeline | LGBMClassifier, features: np.ndarray) -> np.ndarray:
	probs = np.asarray(model.predict_proba(pd.DataFrame(geometric_summary(features))), dtype=np.float64)
	full = np.zeros((len(features), 7))
	full[:, np.asarray(model.classes_, dtype=int)] = probs
	return full


def development_folds(index: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
	"""Forward validation within development weeks only; exclude augmented eval rows."""
	folds = []
	for week in (3, 4, 5):
		train = (index.season.eq(2021) | (index.season.eq(2022) & index.week.lt(week))).to_numpy()
		tune = (index.season.eq(2022) & index.week.eq(week) & ~index.augmented).to_numpy()
		folds.append((np.flatnonzero(train), np.flatnonzero(tune)))
	return folds


def train_baselines(manifest_path: Path, output_dir: Path) -> Path:
	"""Fit on -0.1s training snapshots and report development results only."""
	import json

	import joblib

	from dataset_io import load_examples
	from evaluate import KEYS, PROB_COLUMNS, evaluate_predictions

	x_train, _, y_train, train_index = load_examples(manifest_path, "fit", offset=-1)
	x_tune, _, y_tune, tune_index = load_examples(manifest_path, "tune", offset=-1)
	output_dir.mkdir(parents=True, exist_ok=False)
	manifest = json.loads(manifest_path.read_text())
	receipt = {
		"dataset_hash": manifest["dataset_hash"],
		"endpoint": -1,
		"training_examples": len(x_train),
		"training_plays": int(train_index.loc[~train_index.augmented].shape[0]),
		"tune_examples": len(x_tune),
		"models": {},
	}
	for kind in ("prior", "logistic", "lightgbm"):
		if kind == "prior":
			counts = np.bincount(y_train.numpy()[~train_index.augmented.to_numpy()], minlength=7)
			probs = np.tile(counts / counts.sum(), (len(x_tune), 1))
		else:
			model = fit_baseline(x_train[:, -1].numpy(), y_train.numpy(), kind)
			joblib.dump(model, output_dir / f"{kind}.joblib")
			probs = predict_baseline_proba(model, x_tune[:, -1].numpy())
		predictions = pd.DataFrame(tune_index[KEYS]).copy()
		predictions["target"] = y_tune.numpy()
		predictions[PROB_COLUMNS] = probs
		predictions.to_parquet(output_dir / f"{kind}_tune.parquet", index=False)
		metrics_path = evaluate_predictions(predictions, manifest_path, output_dir / kind)
		receipt["models"][kind] = json.loads(metrics_path.read_text())
	path = output_dir / "baseline-results.json"
	path.write_text(json.dumps(receipt, indent=2) + "\n")
	return path
