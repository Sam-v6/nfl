"""Causal tracking prefixes and reproducible, game-separated dataset shards."""

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from clean_data import calculate_velocity_components, make_plays_left_to_right, rotate_direction_and_orientation
from coverage_labels import CLASSES, canonicalize_coverage

FEATURES = ("x_clean", "y_clean", "v_x", "v_y", "defense")
PREPROCESSING_VERSION = 1


def file_hash(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		while chunk := stream.read(1024 * 1024):
			digest.update(chunk)
	return digest.hexdigest()


def split_for_week(season: int, week: int) -> str:
	if season == 2021 and 1 <= week <= 8:
		return "fit"
	if season == 2022 and 1 <= week <= 9:
		return "fit" if week <= 4 else {5: "tune", 6: "calibrate", 7: "final", 8: "final", 9: "historical"}[week]
	raise ValueError(f"Unsupported season/week: {season}/{week}")


def prepare_prefix(tracking: pd.DataFrame, cutoff_frame: int, defensive_team: str, history_frames: int = 30, feature_names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
	"""Use observed frames only; absent/incomplete history stays zero and masked."""
	features = tuple(feature_names) if feature_names is not None else FEATURES
	if features != FEATURES:
		raise ValueError("Unsupported feature schema; outcome fields are never model inputs")
	if history_frames < 1:
		raise ValueError("history_frames must be positive")
	prefix = tracking.loc[tracking.frameId.between(cutoff_frame - history_frames + 1, cutoff_frame)].copy()
	if "club" not in prefix:
		prefix["club"] = prefix["team"]
	prefix = prefix.loc[prefix.club != "football"].copy()
	if prefix.duplicated(["frameId", "nflId"]).any():
		raise ValueError("Duplicate player/frame key")
	last = prefix.loc[prefix.frameId == cutoff_frame]
	if len(last) != 22 or last.nflId.isna().any() or last.nflId.nunique() != 22 or last.club.eq(defensive_team).sum() != 11:
		raise ValueError("invalid_cutoff")
	ids = sorted(last.nflId.tolist())
	prefix = rotate_direction_and_orientation(prefix)
	prefix = make_plays_left_to_right(prefix)
	prefix = calculate_velocity_components(prefix)
	prefix["defense"] = prefix.club.eq(defensive_team).astype(float)
	cube = np.zeros((history_frames, 22, len(FEATURES)), dtype=np.float32)
	mask = np.zeros(history_frames, dtype=bool)
	for _, frame in prefix.groupby("frameId", sort=False):
		if len(frame) != 22 or set(frame.nflId) != set(ids) or frame.club.eq(defensive_team).sum() != 11:
			continue
		values = frame.set_index("nflId").loc[ids, list(FEATURES)].to_numpy(dtype=np.float32)
		if not np.isfinite(values).all():
			continue
		slot = int(frame.frameId.iloc[0]) - cutoff_frame + history_frames - 1
		cube[slot] = values
		mask[slot] = True
	if not mask[-1]:
		raise ValueError("invalid_cutoff")
	return cube, mask


def prepare_dataset(data_root: Path, config: dict[str, Any], output_dir: Path) -> Path:
	"""Hash raw inputs and emit immutable weekly tensor shards plus keyed index."""
	seasons = config.get("seasons", [2021, 2022])
	cutoffs = config.get("cutoffs", [-30, -20, -10, -5, -1])
	history = int(config.get("history_frames", 30))
	seed = int(config.get("seed", 42))
	if not seasons or len(set(seasons)) != len(seasons) or any(s not in (2021, 2022) for s in seasons):
		raise ValueError("seasons must contain unique supported seasons")
	if not cutoffs or any(not isinstance(c, int) or c >= 0 for c in cutoffs) or len(set(cutoffs)) != len(cutoffs):
		raise ValueError("cutoffs must be unique, strictly pre-snap integer offsets")
	if history < 1 or history > 150:
		raise ValueError("history_frames must be between 1 and 150")
	if output_dir.resolve().is_relative_to(data_root.resolve()):
		raise ValueError("Run directory must be outside read-only data root")
	paths = []
	for season in seasons:
		base = data_root / str(season)
		for name in ("games.parquet", "plays.parquet"):
			path = base / name
			if not path.is_file():
				raise ValueError(f"Missing data file: {path}")
			paths.append(path)
		weekly = sorted(base.glob("tracking_week_*.parquet"))
		if not weekly:
			raise ValueError(f"No tracking data for season {season}")
		paths.extend(weekly)
	inputs = {str(path.relative_to(data_root)): file_hash(path) for path in sorted(paths)}
	settings = {"seasons": seasons, "cutoffs": cutoffs, "history_frames": history, "seed": seed}
	identity = {
		"inputs": inputs,
		"settings": settings,
		"classes": CLASSES,
		"features": FEATURES,
		"preprocessing_version": PREPROCESSING_VERSION,
		"preprocessing_hashes": {name: file_hash(Path(__file__).with_name(name)) for name in ("dataset.py", "clean_data.py", "coverage_labels.py")},
	}
	dataset_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
	output_dir.mkdir(parents=True, exist_ok=False)
	rng = np.random.default_rng(seed)
	index: list[dict[str, Any]] = []
	shards: list[dict[str, Any]] = []
	exclusions: Counter[str] = Counter()
	for season in seasons:
		base = data_root / str(season)
		games = pd.read_parquet(base / "games.parquet")
		plays = pd.read_parquet(base / "plays.parquet")
		plays = plays.merge(games[["gameId", "week"]], on="gameId", how="left", validate="many_to_one")
		if plays.duplicated(["gameId", "playId"]).any() or plays.week.isna().any():
			raise ValueError("Duplicate plays or missing game metadata")
		plays["target"] = plays.pff_passCoverage.map(canonicalize_coverage)
		cohort = plays.passResult.isin(["C", "I", "IN", "S"])
		exclusions["outside_pass_cohort"] += int((~cohort).sum())
		exclusions["unknown_label"] += int((cohort & plays.target.isna()).sum())
		eligible = plays.loc[cohort & plays.target.notna()].set_index(["gameId", "playId"])
		for path in sorted(base.glob("tracking_week_*.parquet")):
			week = int(path.stem.rsplit("_", 1)[-1])
			split = split_for_week(season, week)
			logging.info("Preparing season %s week %s", season, week)
			tracking = pd.read_parquet(path)
			arrays, masks, targets = [], [], []
			shard_path = f"s{season}_w{week}.pt"
			for _, play in tracking.groupby(["gameId", "playId"], sort=True):
				game_id, play_id = int(play.gameId.iloc[0]), int(play.playId.iloc[0])
				if (game_id, play_id) not in eligible.index:
					continue
				metadata = eligible.loc[(game_id, play_id)]
				if int(metadata.week) != week:
					raise ValueError("Tracking week does not match game metadata")
				snaps = play.loc[play.event.eq("ball_snap"), "frameId"].unique()
				if len(snaps) != 1:
					exclusions["missing_or_ambiguous_snap"] += 1
					continue
				for offset in cutoffs:
					cutoff = int(snaps[0]) + offset
					try:
						array, mask = prepare_prefix(play, cutoff, str(metadata.defensiveTeam), history)
					except ValueError as exc:
						exclusions["duplicate_frame" if "Duplicate" in str(exc) else "invalid_cutoff"] += 1
						continue
					key = f"{season}_{int(game_id)}_{int(play_id)}_{cutoff}"
					row = {
						"season": season,
						"gameId": int(game_id),
						"playId": int(play_id),
						"week": week,
						"cutoff_frame": cutoff,
						"offset": offset,
						"split": split,
						"target": int(metadata.target),
						"original_label": str(metadata.pff_passCoverage),
						"observed_history": int(mask.sum()),
						"parent_key": key,
						"augmented": False,
						"shard": shard_path,
						"shard_row": len(arrays),
					}
					index.append(row)
					arrays.append(array)
					masks.append(mask)
					targets.append(int(metadata.target))
					if split == "fit" and rng.random() < 1 / 3:
						mirrored = array.copy()
						mirrored[mask, :, 1] = 160 / 3 - mirrored[mask, :, 1]
						mirrored[mask, :, 3] *= -1
						index.append({**row, "augmented": True, "shard_row": len(arrays)})
						arrays.append(mirrored)
						masks.append(mask)
						targets.append(int(metadata.target))
			if arrays:
				torch.save(
					{"features": torch.from_numpy(np.stack(arrays)), "time_mask": torch.from_numpy(np.stack(masks)), "targets": torch.tensor(targets, dtype=torch.long)}, output_dir / shard_path
				)
				shards.append({"path": shard_path, "rows": len(arrays), "split": split, "season": season, "week": week, "sha256": file_hash(output_dir / shard_path)})
	if not index:
		raise ValueError("No eligible pre-snap examples; inspect exclusions")
	examples = pd.DataFrame(index)
	if examples.groupby(["season", "gameId"])["split"].nunique().max() != 1:
		raise ValueError("Game overlap between splits")
	examples.to_parquet(output_dir / "examples.parquet", index=False)
	manifest = {
		**identity,
		"dataset_hash": dataset_hash,
		"shards": shards,
		"examples_sha256": file_hash(output_dir / "examples.parquet"),
		"exclusions": dict(exclusions),
		"rows_by_split": examples.groupby("split").size().to_dict(),
		"heldout_history": "Weeks 7-8 were used by historical models; reserved from new runs only",
		"status": "complete",
	}
	manifest_path = output_dir / "manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
	return manifest_path
