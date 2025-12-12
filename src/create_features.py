#!/usr/bin/env python
"""
Builds weekly training and validation feature tensors from raw tracking data.
"""

import gc
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from clean_data import (
	add_frames_from_snap,
	calculate_velocity_components,
	data_augmentation,
	label_offense_defense_manzone,
	make_plays_left_to_right,
	pass_attempt_merging,
	prepare_frame_data,
	rotate_direction_and_orientation,
	select_augmented_frames,
)
from common.args import parse_args
from common.decorators import set_time_decorators_enabled, time_fcn
from common.paths import PROCESSED_DIR
from load_data import RawDataLoader


def clean_df(location_df: pd.DataFrame, plays_df: pd.DataFrame, game_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Cleans a week of tracking data and enriches it with play context.

	Inputs:
	- location_df: Raw tracking rows for a given set of plays.
	- plays_df: Play metadata including man/zone labels.
	- game_df: Game metadata containing week numbers.

	Outputs:
	- cleaned_df: Tracking rows with directional fixes, labels, and snap-relative fields.
	"""

	location_df = rotate_direction_and_orientation(location_df)
	location_df = make_plays_left_to_right(location_df)
	location_df = calculate_velocity_components(location_df)
	location_df = pass_attempt_merging(location_df, plays_df)
	location_df = label_offense_defense_manzone(location_df, plays_df)

	location_df = location_df[(location_df["club"] != "football") & (location_df["passAttempt"] == 1)]

	location_df = location_df.merge(game_df[["gameId", "week"]], on="gameId", how="left")

	location_df["uniqueId"] = location_df["gameId"].astype(str) + "_" + location_df["playId"].astype(str)
	location_df["frameUniqueId"] = location_df["gameId"].astype(str) + "_" + location_df["playId"].astype(str) + "_" + location_df["frameId"].astype(str)

	location_df = add_frames_from_snap(location_df)
	location_df = location_df[(location_df["frames_from_snap"] >= -150) & (location_df["frames_from_snap"] <= 50)]

	return location_df


def _process_df(location_df: pd.DataFrame, plays_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Applies cleaning, subsampling, and augmentation to a week's tracking data.

	Inputs:
	- location_df: Raw tracking rows for the week.
	- plays_df: Play-level labels and metadata.
	- games_df: Game-level metadata including weeks.

	Outputs:
	- processed_df: Combined original and augmented rows ready for tensorization.
	"""

	location_df = clean_df(location_df, plays_df, games_df)
	location_df = location_df[location_df["frameId"] % 2 == 0]

	num_unique_frames = len(set(location_df["frameUniqueId"]))
	selected_frames = select_augmented_frames(location_df, int(num_unique_frames / 3), sigma=5)
	augmented_location_df = data_augmentation(location_df, selected_frames)

	combined_location_df = pd.concat([location_df, augmented_location_df])

	return combined_location_df


def _load_weeks(prefix_seasons_weeks: dict[str, Sequence[int]], prefix: str, PROCESSED_DIR: Path) -> torch.Tensor:
	"""
	Loads and concatenates saved tensors for a set of weeks.

	Inputs:
	- prefix_seasons_weeks: map that contains season as key and the weeks in a list as the values.
	- prefix: Base filename prefix (e.g., 'features').
	- PROCESSED_DIR: Directory where tensors are stored.

	Outputs:
	- tensor: Concatenated tensor spanning the requested weeks.
	"""

	tensors = []
	for s in prefix_seasons_weeks.keys():
		for w in prefix_seasons_weeks[s]:
			tensors.append(torch.load(PROCESSED_DIR / f"s{s}_w{w}_{prefix}.pt"))
	return torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]


@time_fcn
def main() -> None:
	"""
	Generates and saves weekly and pooled tensors for transformer training.

	Inputs:
	- Command-line flags control profiling.

	Outputs:
	- Serialized feature and target tensors for training and validation weeks.
	"""

	# Enable logging
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

	# Parse command line args
	args = parse_args()
	if args.profile:
		set_time_decorators_enabled(True)
		logging.info("Timing decorators enabled")
	else:
		set_time_decorators_enabled(False)
		logging.info("Timing decorators disabled")

	# Define training/validation
	seasons_weeks = {"2021": range(1, 9), "2022": range(1, 10)}

	# Features and target defintions
	features = ["x_clean", "y_clean", "v_x", "v_y", "defense"]
	target_column = "pff_manZone"

	# Create dataloader and get the main data
	raw = RawDataLoader()
	games_df, plays_df, players_df = raw.get_base_data()

	for s in seasons_weeks.keys():
		for w in seasons_weeks[s]:
			logging.info(f"Processing week: {w}")
			location_df_w = raw.get_tracking_data(weeks=[w], seasons=[s])
			combined_loc_df_w = _process_df(location_df_w, plays_df, games_df)

			keep_cols = ["frameUniqueId", "displayName", "frameId", "frameType", "x_clean", "y_clean", "v_x", "v_y", "defensiveTeam", "pff_manZone", "defense"]
			combined_loc_df_w = combined_loc_df_w[keep_cols].copy()

			for c in ["x_clean", "y_clean", "v_x", "v_y", "defense"]:
				combined_loc_df_w[c] = pd.to_numeric(combined_loc_df_w[c], downcast="float")

			features_tensor_w, targets_tensor_w = prepare_frame_data(combined_loc_df_w, features=features, target_column=target_column)

			torch.save(features_tensor_w, PROCESSED_DIR / f"s{s}_w{w}_features.pt")
			torch.save(targets_tensor_w, PROCESSED_DIR / f"s{s}_w{w}_targets.pt")

			del location_df_w, combined_loc_df_w, features_tensor_w, targets_tensor_w
			gc.collect()

	# Splitting out training and validation
	train_seasons_weeks = {"2021": range(1, 9), "2022": range(1, 9)}
	val_seasons_weeks = {"2022": [9]}

	logging.info("Aggregrating training set")
	train_features = _load_weeks(train_seasons_weeks, "features", PROCESSED_DIR)
	train_targets = _load_weeks(train_seasons_weeks, "targets", PROCESSED_DIR)

	logging.info("Aggregrating validation set")
	val_features = _load_weeks(val_seasons_weeks, "features", PROCESSED_DIR)
	val_targets = _load_weeks(val_seasons_weeks, "targets", PROCESSED_DIR)

	train_features = train_features.to(torch.float32)
	val_features = val_features.to(torch.float32)

	torch.save(train_features, PROCESSED_DIR / "features_training.pt")
	torch.save(train_targets, PROCESSED_DIR / "targets_training.pt")
	torch.save(val_features, PROCESSED_DIR / "features_val.pt")
	torch.save(val_targets, PROCESSED_DIR / "targets_val.pt")

	logging.info("Train features: %s", getattr(train_features, "shape", None))
	logging.info("Val features:   %s", getattr(val_features, "shape", None))
	logging.info("Train targets:  %s", getattr(train_targets, "shape", None))
	logging.info("Val targets:    %s", getattr(val_targets, "shape", None))


if __name__ == "__main__":
	main()
