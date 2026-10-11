import importlib
from pathlib import Path

import numpy as np


def test_geometry_summary_is_player_permutation_invariant() -> None:
	module = importlib.import_module("models.baselines")
	x = np.zeros((2, 22, 5), dtype=np.float32)
	x[:, :11, 4] = 1
	x[:, :, 0] = np.arange(22)
	x[:, :, 1] = np.arange(22)[::-1]
	permutation = np.random.default_rng(42).permutation(22)
	np.testing.assert_allclose(module.geometric_summary(x), module.geometric_summary(x[:, permutation]))


def test_baseline_uses_training_classes_but_emits_full_vocabulary() -> None:
	module = importlib.import_module("models.baselines")
	x = np.zeros((20, 22, 5), dtype=np.float32)
	x[:, :11, 4] = 1
	x[10:, :11, 0] = 10
	y = np.array([3] * 10 + [5] * 10)
	model = module.fit_baseline(x, y, "logistic")
	probs = module.predict_baseline_proba(model, x)
	assert probs.shape == (20, 7)
	np.testing.assert_allclose(probs.sum(axis=1), 1.0)
	assert np.array_equal(probs.argmax(axis=1), y)
	assert not probs[:, [0, 1, 2, 4, 6]].any()


def test_forward_folds_never_train_on_later_games() -> None:
	import pandas as pd

	module = importlib.import_module("models.baselines")
	index = pd.DataFrame({"season": [2022] * 9, "week": range(1, 10), "augmented": [False] * 9})
	folds = module.development_folds(index)
	assert len(folds) == 3
	for train_rows, tune_rows in folds:
		assert index.loc[train_rows, "week"].max() < index.loc[tune_rows, "week"].min()
		assert index.loc[tune_rows, "week"].max() <= 5
		assert not set(train_rows) & set(tune_rows)


def test_baseline_run_uses_development_data_and_writes_keyed_report(tmp_path: Path) -> None:
	import json

	import pandas as pd
	from test_dataset import raw_fixture

	from dataset import prepare_dataset

	raw = tmp_path / "raw"
	raw_fixture(raw)
	plays = pd.read_parquet(raw / "2022/plays.parquet")
	pd.concat([plays, plays.iloc[[0]].assign(playId=11, pff_passCoverage="2-Man")]).to_parquet(raw / "2022/plays.parquet", index=False)
	tracking = pd.read_parquet(raw / "2022/tracking_week_1.parquet")
	pd.concat([tracking, tracking.assign(playId=11, x=tracking.x + 10)]).to_parquet(raw / "2022/tracking_week_1.parquet", index=False)
	manifest = prepare_dataset(raw, {"seasons": [2022], "cutoffs": [-1]}, tmp_path / "prepared")
	result = importlib.import_module("models.baselines").train_baselines(manifest, tmp_path / "baselines")
	report = json.loads(result.read_text())
	assert report["training_plays"] == 2
	assert report["tune_examples"] == 1
	assert set(report["models"]) == {"prior", "logistic", "lightgbm"}
	assert not (manifest.parent / "selection.json").exists()
	assert pd.read_parquet(result.parent / "logistic_tune.parquet").gameId.tolist() == [2]
