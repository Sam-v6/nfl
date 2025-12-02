#!/usr/bin/env python
"""
Cleans and prepares tracking data for downstream modeling and feature creation.
"""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


def rotate_direction_and_orientation(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Rotates direction and orientation so field left-to-right is 0 degrees.

	Inputs:
	- df: Tracking rows containing original direction and orientation.

	Outputs:
	- rotated_df: Dataframe with normalized direction/orientation columns.
	"""

	df["o_clean"] = (-(df["o"] - 90)) % 360
	df["dir_clean"] = (-(df["dir"] - 90)) % 360
	return df


def make_plays_left_to_right(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Flips plays so every snap proceeds left to right and creates cleaned columns.

	Inputs:
	- df: Tracking rows with original play direction.

	Outputs:
	- standardized_df: Dataframe with *_clean columns aligned left to right.
	"""

	df["x_clean"] = np.where(
		df["playDirection"] == "left",
		120 - df["x"],
		df["x"],
	)

	df["y_clean"] = df["y"]
	df["s_clean"] = df["s"]
	df["a_clean"] = df["a"]
	df["dis_clean"] = df["dis"]

	df["o_clean"] = np.where(df["playDirection"] == "left", 180 - df["o_clean"], df["o_clean"])

	df["o_clean"] = (df["o_clean"] + 360) % 360

	df["dir_clean"] = np.where(df["playDirection"] == "left", 180 - df["dir_clean"], df["dir_clean"])

	df["dir_clean"] = (df["dir_clean"] + 360) % 360

	return df


def calculate_velocity_components(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Derives velocity components from cleaned direction and speed.

	Inputs:
	- df: Tracking rows with normalized direction and speed columns.

	Outputs:
	- velocity_df: Dataframe with v_x and v_y components added.
	"""
	df["dir_radians"] = np.radians(df["dir_clean"])
	df["v_x"] = df["s_clean"] * np.cos(df["dir_radians"])
	df["v_y"] = df["s_clean"] * np.sin(df["dir_radians"])
	return df


def label_offense_defense_coverage(presnap_df: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Maps PFF coverage strings to numeric labels and marks defenders.

	Inputs:
	- presnap_df: Tracking rows before the snap.
	- plays_df: Play metadata including pass coverage labels.

	Outputs:
	- labeled_df: Presnap rows with defense flag and coverage label added.
	"""
	coverage_replacements = {
		"Cover-3 Cloud Right": "Cover-3",
		"Cover-3 Cloud Left": "Cover-3",
		"Cover-3 Seam": "Cover-3",
		"Cover-3 Double Cloud": "Cover-3",
		"Cover-6 Right": "Cover-6",
		"Cover 6-Left": "Cover-6",
		"Cover-1 Double": "Cover-1",
	}

	values_to_drop = ["Miscellaneous", "Bracket", "Prevent", "Red Zone", "Goal Line"]

	plays_df["pff_passCoverage"] = plays_df["pff_passCoverage"].replace(coverage_replacements)

	plays_df = plays_df.dropna(subset=["pff_passCoverage"])
	plays_df = plays_df[~plays_df["pff_passCoverage"].isin(values_to_drop)]

	coverage_mapping = {"Cover-0": 0, "Cover-1": 1, "Cover-2": 2, "Cover-3": 3, "Quarters": 4, "2-Man": 5, "Cover-6": 6}

	merged_df = presnap_df.merge(plays_df[["gameId", "playId", "possessionTeam", "defensiveTeam", "pff_passCoverage"]], on=["gameId", "playId"], how="left")

	merged_df["defense"] = ((merged_df["club"] == merged_df["defensiveTeam"]) & (merged_df["club"] != "football")).astype(int)

	merged_df["pff_passCoverage"] = merged_df["pff_passCoverage"].map(coverage_mapping)
	merged_df.dropna(subset=["pff_passCoverage"], inplace=True)

	return merged_df


def label_offense_defense_manzone(presnap_df: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Adds man/zone numeric labels and defense flags to presnap rows.

	Inputs:
	- presnap_df: Tracking rows before the snap.
	- plays_df: Play metadata including man/zone tags.

	Outputs:
	- labeled_df: Presnap rows with defense flag and man/zone label added.
	"""
	plays_df = plays_df.dropna(subset=["pff_manZone"])

	coverage_mapping = {"Zone": 0, "Man": 1}

	merged_df = presnap_df.merge(plays_df[["gameId", "playId", "possessionTeam", "defensiveTeam", "pff_manZone"]], on=["gameId", "playId"], how="left")

	merged_df["defense"] = ((merged_df["club"] == merged_df["defensiveTeam"]) & (merged_df["club"] != "football")).astype(int)

	merged_df["pff_manZone"] = merged_df["pff_manZone"].map(coverage_mapping)
	merged_df.dropna(subset=["pff_manZone"], inplace=True)

	return merged_df


def label_offense_defense_formation(presnap_df: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Attaches offense/defense flags and offensive formations to presnap rows.

	Inputs:
	- presnap_df: Tracking rows before the snap.
	- plays_df: Play metadata including formation.

	Outputs:
	- labeled_df: Presnap rows with formation code and offense/defense indicators.
	"""
	formation_mapping = {"EMPTY": 0, "I_FORM": 1, "JUMBO": 2, "PISTOL": 3, "SHOTGUN": 4, "SINGLEBACK": 5, "WILDCAT": 6}

	merged_df = presnap_df.merge(plays_df[["gameId", "playId", "possessionTeam", "defensiveTeam", "offenseFormation"]], on=["gameId", "playId"], how="left")

	merged_df["defense"] = ((merged_df["club"] == merged_df["defensiveTeam"]) & (merged_df["club"] != "football")).astype(int)

	merged_df["offenseFormation"] = merged_df["offenseFormation"].map(formation_mapping)
	merged_df.dropna(subset=["offenseFormation"], inplace=True)

	return merged_df


def split_data_by_uniqueId(
	df: pd.DataFrame,
	train_ratio: float = 0.7,
	test_ratio: float = 0.15,
	val_ratio: float = 0.15,
	unique_id_column: str = "uniqueId",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Splits tracking rows into train/test/val while keeping plays intact.

	Inputs:
	- df: Full set of tracking rows.
	- train_ratio/test_ratio/val_ratio: Fractions used for each split.
	- unique_id_column: Column defining play identity.

	Outputs:
	- splits: Three dataframes for train, test, and validation.
	"""
	unique_ids = df[unique_id_column].unique()
	np.random.shuffle(unique_ids)

	num_ids = len(unique_ids)
	train_end = int(train_ratio * num_ids)
	test_end = train_end + int(test_ratio * num_ids)

	train_ids = unique_ids[:train_end]
	test_ids = unique_ids[train_end:test_end]
	val_ids = unique_ids[test_end:]

	train_df = df[df[unique_id_column].isin(train_ids)]
	test_df = df[df[unique_id_column].isin(test_ids)]
	val_df = df[df[unique_id_column].isin(val_ids)]

	print(f"Train Dataframe Frames: {train_df.shape[0]}")
	print(f"Test Dataframe Frames: {test_df.shape[0]}")
	print(f"Val Dataframe Frames: {val_df.shape[0]}")

	return train_df, test_df, val_df


def pass_attempt_merging(tracking: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
	"""
	Flags pass attempts and merges that indicator into tracking rows.

	Inputs:
	- tracking: Tracking rows to annotate.
	- plays: Play metadata with pass results.

	Outputs:
	- merged_df: Tracking rows with passAttempt column.
	"""
	plays["passAttempt"] = np.where(plays["passResult"].isin(["C", "I", "IN", "S"]), 1, 0)

	plays_for_merge = plays[["gameId", "playId", "passAttempt"]]
	merged_df = tracking.merge(plays_for_merge, on=["gameId", "playId"], how="left")

	return merged_df


def prepare_frame_data(df: pd.DataFrame, features: Sequence[str], target_column: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
	"""
	Converts per-frame tracking rows into stacked feature and target tensors.

	Inputs:
	- df: Frame-level tracking rows with required features.
	- features: Column names to include as model inputs.
	- target_column: Column containing frame target labels.

	Outputs:
	- feature_tensor: Batched tensor of frame features.
	- target_tensor: Tensor of frame targets aligned to feature order.
	"""
	features_array = df.groupby("frameUniqueId")[features].apply(lambda x: x.to_numpy(dtype=np.float32)).to_numpy()

	try:
		features_tensor = torch.tensor(np.stack(features_array))
	except ValueError as e:
		print("Skipping batch due to inconsistent shapes in features_array:", e)
		return None, None

	targets_array = df.groupby("frameUniqueId")[target_column].first().to_numpy()
	targets_tensor = torch.tensor(targets_array, dtype=torch.long)

	return features_tensor, targets_tensor


def select_augmented_frames(df: pd.DataFrame, num_samples: int, sigma: int = 5) -> np.ndarray:
	"""
	Samples frames for augmentation, biased around the snap.

	Inputs:
	- df: Tracking rows with frame identifiers and snap offsets.
	- num_samples: Number of frame identifiers to sample.
	- sigma: Spread of the snap-centered weighting.

	Outputs:
	- selected_frames: Array of frameUniqueId values to augment.
	"""
	df_frames = df[["frameUniqueId", "frames_from_snap"]].drop_duplicates()
	weights = np.exp(-((df_frames["frames_from_snap"] + 10) ** 2) / (2 * sigma**2))

	weights /= weights.sum()

	selected_frames = np.random.choice(df_frames["frameUniqueId"], size=num_samples, replace=False, p=weights)

	return selected_frames


def data_augmentation(df: pd.DataFrame, augmented_frames: Iterable[str]) -> pd.DataFrame:
	"""
	Mirrors selected frames to add variation around the snap.

	Inputs:
	- df: Tracking rows ready for augmentation.
	- augmented_frames: Frame identifiers chosen for mirroring.

	Outputs:
	- augmented_df: Augmented rows with flipped coordinates and tags.
	"""
	df_sample = df.loc[df["frameUniqueId"].isin(augmented_frames)].copy()

	df_sample["y_clean"] = (160 / 3) - df_sample["y_clean"]
	df_sample["dir_radians"] = (2 * np.pi) - df_sample["dir_radians"]
	df_sample["dir_clean"] = np.degrees(df_sample["dir_radians"])

	df_sample["frameUniqueId"] = df_sample["frameUniqueId"].astype(str) + "_aug"

	return df_sample


def add_frames_from_snap(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Calculates snap-relative frame offsets for each play.

	Inputs:
	- df: Tracking rows containing frameType and frameId.

	Outputs:
	- df_with_offsets: Dataframe including frames_from_snap column.
	"""
	snap_frames = df[df["frameType"] == "SNAP"].groupby("uniqueId")["frameId"].first()
	df = df.merge(snap_frames.rename("snap_frame"), on="uniqueId", how="left")
	df["frames_from_snap"] = df["frameId"] - df["snap_frame"]

	return df
