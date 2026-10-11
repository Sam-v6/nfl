import importlib
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


def module() -> ModuleType:
	return importlib.import_module("evaluate")


def predictions(targets: list[int]) -> pd.DataFrame:
	frame = pd.DataFrame({"season": 2022, "gameId": [i // 2 for i in range(len(targets))], "playId": range(len(targets)), "cutoff_frame": 99, "target": targets})
	for i in range(7):
		frame[f"prob_{i}"] = np.array(targets) == i
	return frame


def test_perfect_observed_class_does_not_hide_missing_classes() -> None:
	result = module().score_predictions(np.array([3, 3]), np.eye(7)[[3, 3]])
	assert result["accuracy"] == 1.0
	assert result["macro_f1"] == pytest.approx(1 / 7)
	assert result["log_loss"] < 1e-10
	assert result["brier"] == 0.0
	assert result["per_class"]["2-Man"]["support"] == 0
	assert not result["per_class"]["2-Man"]["sufficient_support"]


def test_rare_class_recall_is_visible_despite_high_accuracy() -> None:
	targets = np.array([3] * 99 + [5])
	result = module().score_predictions(targets, np.eye(7)[[3] * 100])
	assert result["accuracy"] == 0.99
	assert result["per_class"]["2-Man"]["recall"] == 0.0
	assert result["macro_f1"] < 0.15


@pytest.mark.parametrize("bad", [np.ones((2, 7)), np.full((2, 7), np.nan), np.eye(7)[[0, 1]] - 0.1])
def test_invalid_probabilities_fail(bad: np.ndarray) -> None:
	with pytest.raises(ValueError, match="probabilit"):
		module().score_predictions(np.array([0, 1]), bad)


def test_binary_probability_is_sum_of_all_man_classes() -> None:
	probabilities = np.array([[0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1]])
	result = module().score_predictions(np.array([5]), probabilities)
	assert result["binary"]["accuracy"] == 1.0
	assert result["binary"]["log_loss"] == pytest.approx(-np.log(0.6))


def test_paired_bootstrap_rejects_different_plays() -> None:
	left = predictions([0, 1, 2, 3])
	right = left.copy()
	right.loc[0, "playId"] = 99
	with pytest.raises(ValueError, match="keys"):
		module().compare_predictions(left, right)


def test_paired_bootstrap_is_game_level_and_reproducible() -> None:
	candidate = predictions([0, 1, 2, 3, 4, 5, 6, 0])
	baseline = candidate.copy()
	for i in range(7):
		baseline[f"prob_{i}"] = 1.0 if i == 3 else 0.0
	one = module().compare_predictions(candidate, baseline)
	two = module().compare_predictions(candidate, baseline)
	assert one == two
	assert one["n_games"] == 4
	assert one["macro_f1_delta"] > 0.8
	assert one["ci95"][0] > 0.0


def manifest_for(tmp_path: Path, frame: pd.DataFrame, split: str) -> Path:
	import hashlib

	from coverage_labels import CLASSES

	index = frame.drop(columns=[f"prob_{i}" for i in range(7)]).assign(split=split, offset=-1, augmented=False)
	index.to_parquet(tmp_path / "examples.parquet", index=False)
	path = tmp_path / "manifest.json"
	path.write_text(json.dumps({"classes": list(CLASSES), "dataset_hash": "frozen-fixture", "examples_sha256": hashlib.sha256((tmp_path / "examples.parquet").read_bytes()).hexdigest()}))
	return path


def test_report_rejects_duplicate_and_missing_predictions(tmp_path: Path) -> None:
	frame = predictions([0, 1, 2, 3])
	manifest = manifest_for(tmp_path, frame, "tune")
	with pytest.raises(ValueError, match="Duplicate"):
		module().evaluate_predictions(pd.concat([frame, frame.iloc[[0]]]), manifest, tmp_path / "report")
	with pytest.raises(ValueError, match="Missing"):
		module().evaluate_predictions(frame.iloc[1:], manifest, tmp_path / "report")
	assert not (tmp_path / "report").exists()


def test_final_evaluation_requires_frozen_selection(tmp_path: Path) -> None:
	frame = predictions([0, 1, 2, 3])
	manifest = manifest_for(tmp_path, frame, "final")
	with pytest.raises(ValueError, match="selection"):
		module().evaluate_predictions(frame, manifest, tmp_path / "report")
	assert not (tmp_path / "report").exists()


def test_report_retains_abstentions_in_denominator(tmp_path: Path) -> None:
	frame = predictions([0, 1, 2, 3])
	manifest = manifest_for(tmp_path, frame, "tune")
	frame["answered"] = [True, True, False, False]
	result = module().evaluate_predictions(frame, manifest, tmp_path / "report")
	report = json.loads(result.read_text())
	assert report["tune/-1"]["eligible_plays"] == 4
	assert report["tune/-1"]["answer_fraction"] == 0.5
	assert report["tune/-1"]["answered_accuracy"] == 1.0
	assert (result.parent / "confusion_tune_-1.png").exists()
