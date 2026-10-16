import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_training import training_fixture


def test_play_weights_equalize_parent_plays() -> None:
	from training import play_sampling_weights

	index = pd.DataFrame({"season": [2021] * 4 + [2022], "gameId": [1] * 5, "playId": [1, 1, 1, 2, 1]})
	weights = play_sampling_weights(index)
	torch.testing.assert_close(weights, torch.tensor([1 / 3, 1 / 3, 1 / 3, 1, 1], dtype=torch.double))


def test_fold_views_exclude_future_and_augmentation(tmp_path: Path) -> None:
	from dataset import file_hash
	from development import build_fold

	source = training_fixture(tmp_path / "data")
	index = pd.read_parquet(source.parent / "examples.parquet")
	index["week"] = index.playId % 6 + 1
	index["gameId"] = index.week
	index.loc[index.playId.eq(0), "season"] = 2021
	index.loc[index.playId.eq(3), "augmented"] = True
	index.to_parquet(source.parent / "examples.parquet", index=False)
	manifest = json.loads(source.read_text())
	manifest["examples_sha256"] = file_hash(source.parent / "examples.parquet")
	source.write_text(json.dumps(manifest))
	fold = build_fold(source, tmp_path / "fold", 3)
	rows = pd.read_parquet(fold.parent / "examples.parquet")
	assert rows.loc[rows.season.eq(2022), "week"].max() == 4
	assert set(rows.loc[rows.split.eq("tune"), "week"]) == {3}
	assert set(rows.loc[rows.split.eq("development"), "week"]) == {4}
	assert not rows.loc[rows.split.ne("fit"), "augmented"].any()
	assert rows.groupby(["season", "gameId", "playId"]).split.nunique().max() == 1
	assert all(Path(p).is_absolute() for p in rows.shard)
	assert json.loads(fold.read_text())["source_manifest_sha256"] == file_hash(source)
	with pytest.raises(FileExistsError):
		build_fold(source, tmp_path / "fold", 3)
	source.write_text(source.read_text().replace(manifest["examples_sha256"], "bad"))
	with pytest.raises(ValueError, match="hash"):
		build_fold(source, tmp_path / "bad", 3)


def test_nested_learning_curve_views_keep_identical_validation(tmp_path: Path) -> None:
	from dataset import file_hash
	from development import build_fold

	source = training_fixture(tmp_path / "data")
	index = pd.read_parquet(source.parent / "examples.parquet")
	index["week"] = index.playId % 4 + 1
	index["gameId"] = index.playId
	index.to_parquet(source.parent / "examples.parquet", index=False)
	metadata = json.loads(source.read_text())
	metadata["examples_sha256"] = file_hash(source.parent / "examples.parquet")
	source.write_text(json.dumps(metadata))
	fits = []
	validation = []
	for fraction in (0.25, 0.5, 1.0):
		path = build_fold(source, tmp_path / str(fraction), 3, fraction)
		rows = pd.read_parquet(path.parent / "examples.parquet")
		fits.append(set(rows.loc[rows.split.eq("fit"), "gameId"]))
		validation.append(rows.loc[rows.split.ne("fit")].reset_index(drop=True))
	assert fits[0] < fits[1] < fits[2]
	pd.testing.assert_frame_equal(validation[0], validation[1])
	pd.testing.assert_frame_equal(validation[1], validation[2])
