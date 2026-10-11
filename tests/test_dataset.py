import importlib
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


def frames(count: int = 5) -> pd.DataFrame:
	return pd.DataFrame(
		[
			{
				"frameId": frame,
				"nflId": player,
				"club": "DEF" if player < 11 else "OFF",
				"x": float(player),
				"y": float(player),
				"s": 2.0,
				"a": 0.0,
				"dis": 0.2,
				"dir": 0.0,
				"o": 0.0,
				"playDirection": "right",
			}
			for frame in range(100 - count + 1, 101)
			for player in range(22)
		]
	)


def dataset_module() -> ModuleType:
	return importlib.import_module("dataset")


def test_future_frames_cannot_change_prefix() -> None:
	module = dataset_module()
	past = frames()
	future = past.loc[past.frameId == 100].assign(frameId=101, x=999.0)
	a, mask = module.prepare_prefix(past, 100, "DEF")
	b, other_mask = module.prepare_prefix(pd.concat([past, future]), 100, "DEF")
	np.testing.assert_array_equal(a, b)
	np.testing.assert_array_equal(mask, other_mask)
	assert mask.sum() == 5
	assert not a[:25].any()
	assert mask[-1]


def test_gap_is_masked_without_backward_fill() -> None:
	module = dataset_module()
	tracking = frames().query("frameId != 98")
	features, mask = module.prepare_prefix(tracking, 100, "DEF")
	assert not mask[-3]
	assert not features[-3].any()
	assert mask.sum() == 4


@pytest.mark.parametrize("defect", ["duplicate", "missing", "nan", "wrong_side", "missing_cutoff"])
def test_bad_cutoff_is_rejected(defect: str) -> None:
	tracking = frames()
	if defect == "duplicate":
		tracking = pd.concat([tracking, tracking.iloc[[-1]]])
	elif defect == "missing":
		tracking = tracking.drop(tracking.index[-1])
	elif defect == "nan":
		tracking.loc[tracking.index[-1], "x"] = np.nan
	elif defect == "wrong_side":
		tracking.loc[tracking.nflId == 11, "club"] = "DEF"
	else:
		tracking = tracking.query("frameId != 100")
	with pytest.raises(ValueError):
		dataset_module().prepare_prefix(tracking, 100, "DEF")


def test_outcome_feature_is_rejected() -> None:
	with pytest.raises(ValueError, match="feature"):
		dataset_module().prepare_prefix(frames(), 100, "DEF", feature_names=["targetX"])


def test_prefix_does_not_mutate_input() -> None:
	tracking = frames()
	original = tracking.copy(deep=True)
	dataset_module().prepare_prefix(tracking, 100, "DEF")
	pd.testing.assert_frame_equal(tracking, original)


def test_reflection_updates_velocity_and_is_reversible() -> None:
	from clean_data import data_augmentation

	tracking = pd.DataFrame({"frameUniqueId": ["a"], "y_clean": [10.0], "dir_radians": [0.5], "v_x": [2.0], "v_y": [3.0]})
	mirrored = data_augmentation(tracking, ["a"])
	assert mirrored.v_y.iloc[0] == -3.0
	assert mirrored.v_x.iloc[0] == 2.0
	restored = data_augmentation(mirrored, ["a_aug"])
	np.testing.assert_allclose(restored[["y_clean", "v_x", "v_y"]].values, [[10, 2, 3]])


def raw_fixture(root: Path) -> None:
	season = root / "2022"
	season.mkdir(parents=True)
	games = pd.DataFrame({"gameId": [1, 2, 3, 4, 5, 6], "week": [1, 5, 6, 7, 8, 9]})
	games.to_parquet(season / "games.parquet", index=False)
	plays = pd.DataFrame([{"gameId": game, "playId": 10, "passResult": "C", "pff_passCoverage": "Cover-3 Seam", "defensiveTeam": "DEF"} for game in games.gameId])
	plays.to_parquet(season / "plays.parquet", index=False)
	for game, week in zip(games.gameId, games.week, strict=True):
		tracking = frames().assign(gameId=game, playId=10, event=None)
		tracking.loc[tracking.frameId == 100, "event"] = "ball_snap"
		tracking.to_parquet(season / f"tracking_week_{week}.parquet", index=False)


def test_preparation_is_deterministic_and_games_stay_in_one_split(tmp_path: Path) -> None:
	module = dataset_module()
	raw = tmp_path / "raw"
	raw_fixture(raw)
	config = {"seasons": [2022], "cutoffs": [-1], "history_frames": 30, "seed": 42}
	first = module.prepare_dataset(raw, config, tmp_path / "one")
	second = module.prepare_dataset(raw, config, tmp_path / "two")
	a, b = json.loads(first.read_text()), json.loads(second.read_text())
	assert a["dataset_hash"] == b["dataset_hash"]
	index = pd.read_parquet(first.parent / "examples.parquet")
	assert index.groupby("gameId")["split"].nunique().max() == 1
	assert dict(zip(index.loc[~index.augmented, "gameId"], index.loc[~index.augmented, "split"], strict=True)) == {1: "fit", 2: "tune", 3: "calibrate", 4: "final", 5: "final", 6: "historical"}
	assert index.loc[index.augmented, "split"].eq("fit").all()
	assert index.loc[~index.augmented, "target"].eq(3).all()
	assert a["classes"] == ["Cover-0", "Cover-1", "Cover-2", "Cover-3", "Quarters", "2-Man", "Cover-6"]

	import torch

	for left, right in zip(a["shards"], b["shards"], strict=True):
		x = torch.load(first.parent / left["path"], weights_only=True)
		y = torch.load(second.parent / right["path"], weights_only=True)
		assert torch.equal(x["features"], y["features"])
		assert x["features"].dtype == torch.float32
		assert x["time_mask"].dtype == torch.bool


def test_input_changes_invalidate_dataset_hash(tmp_path: Path) -> None:
	module = dataset_module()
	raw = tmp_path / "raw"
	raw_fixture(raw)
	config = {"seasons": [2022], "cutoffs": [-1]}
	first = module.prepare_dataset(raw, config, tmp_path / "one")
	plays = pd.read_parquet(raw / "2022/plays.parquet")
	plays.loc[0, "pff_passCoverage"] = "Cover-2"
	plays.to_parquet(raw / "2022/plays.parquet", index=False)
	second = module.prepare_dataset(raw, config, tmp_path / "two")
	assert json.loads(first.read_text())["dataset_hash"] != json.loads(second.read_text())["dataset_hash"]


def test_bad_frame_does_not_discard_other_plays(tmp_path: Path) -> None:
	module = dataset_module()
	raw = tmp_path / "raw"
	raw_fixture(raw)
	path = raw / "2022/tracking_week_1.parquet"
	tracking = pd.read_parquet(path)
	bad = tracking.assign(playId=11)
	bad = bad.drop(bad.index[-23])
	pd.concat([tracking, bad]).to_parquet(path, index=False)
	plays = pd.read_parquet(raw / "2022/plays.parquet")
	pd.concat([plays, plays.iloc[[0]].assign(playId=11)]).to_parquet(raw / "2022/plays.parquet", index=False)
	manifest = module.prepare_dataset(raw, {"seasons": [2022], "cutoffs": [-1]}, tmp_path / "run")
	index = pd.read_parquet(manifest.parent / "examples.parquet")
	assert index.loc[index.gameId == 1, "playId"].eq(10).all()
	assert json.loads(manifest.read_text())["exclusions"]["invalid_cutoff"] == 1


def test_audit_retains_original_and_canonical_labels(tmp_path: Path) -> None:
	from audit_data import audit_data

	raw = tmp_path / "raw"
	raw_fixture(raw)
	report = audit_data(raw, tmp_path / "audit")
	inventory = pd.read_parquet(report.parent / "plays.parquet")
	assert inventory.original_label.eq("Cover-3 Seam").all()
	assert inventory.target.eq(3).all()
	assert len(inventory) == 6
	assert json.loads(report.read_text())["coverage_counts"]["Cover-3"] == 6


def test_missing_snap_and_unknown_labels_are_receipted(tmp_path: Path) -> None:
	module = dataset_module()
	raw = tmp_path / "raw"
	raw_fixture(raw)
	plays = pd.read_parquet(raw / "2022/plays.parquet")
	plays.loc[plays.gameId == 2, "pff_passCoverage"] = "Bracket"
	plays.to_parquet(raw / "2022/plays.parquet", index=False)
	tracking = pd.read_parquet(raw / "2022/tracking_week_6.parquet").assign(event=None)
	tracking.to_parquet(raw / "2022/tracking_week_6.parquet", index=False)
	manifest = module.prepare_dataset(raw, {"seasons": [2022], "cutoffs": [-1]}, tmp_path / "run")
	report = json.loads(manifest.read_text())
	assert report["exclusions"]["unknown_label"] == 1
	assert report["exclusions"]["missing_or_ambiguous_snap"] == 1
	index = pd.read_parquet(manifest.parent / "examples.parquet")
	assert not index.gameId.isin([2, 3]).any()


@pytest.mark.parametrize("cutoffs", [[0], [1], [-1, -1]])
def test_invalid_cutoffs_fail_before_creating_output(cutoffs: list[int], tmp_path: Path) -> None:
	with pytest.raises(ValueError, match="pre-snap"):
		dataset_module().prepare_dataset(tmp_path / "raw", {"cutoffs": cutoffs}, tmp_path / "run")
	assert not (tmp_path / "run").exists()
