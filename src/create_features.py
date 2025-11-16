#!/usr/bin/env python

"""
Creates features for model training

Requires:
- Raw location tracking data in NFL_HOME/data/parquet

Cleans data and creates the following in NFL_HOME/data/processed:
- features_training.pt
- features_val.pt
- targets_training.pt
- targets_val.pt
"""

import gc
import os
import math
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from load_data import RawDataLoader
from clean_data import *
from common.decorators import time_fcn

def clean_df(location_df: pd.DataFrame, plays_df: pd.DataFrame, game_df: pd.DataFrame):
    
    # Clean data
    location_df = rotate_direction_and_orientation(location_df)
    location_df = make_plays_left_to_right(location_df)
    location_df = calculate_velocity_components(location_df)
    location_df = pass_attempt_merging(location_df, plays_df)
    # location_df = label_offense_defense_coverage(location_df, plays_df)  # for specific coverage... currently set to man/zone only
    location_df = label_offense_defense_manzone(location_df, plays_df)

    # Filter out the football and if it's not a pass attempt
    location_df = location_df[(location_df['club'] != 'football') & (location_df['passAttempt'] == 1)]

    # Add in the week number
    location_df = location_df.merge(
      game_df[["gameId", "week"]],
      on="gameId",
      how="left"
    )

    # Create uids
    location_df['uniqueId'] = location_df['gameId'].astype(str) + "_" + location_df['playId'].astype(str)
    location_df['frameUniqueId'] = (
        location_df['gameId'].astype(str) + "_" +
        location_df['playId'].astype(str) + "_" +
        location_df['frameId'].astype(str))

    # Adding frames_from_snap (to do: make this a function but fine for now)
    location_df = add_frames_from_snap(location_df)

    # Get rid of noisier outliers out of scope (15 seconds after the snap)
    location_df = location_df[(location_df['frames_from_snap'] >= -150) & (location_df['frames_from_snap'] <= 50)]

    return location_df


def _process_df(location_df: pd.DataFrame, plays_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
  
  location_df = clean_df(location_df, plays_df, games_df)

  # Filtering only for even frames (reduce amount of data that looks the same)
  location_df = location_df[location_df['frameId'] % 2 == 0]

  # Apply data augmentation to increase training size (centered around 0-4 seconds presnap!)
  # -- 1/3rd of the current num of frames... specifically selecting for frames around the snap
  num_unique_frames = len(set(location_df['frameUniqueId']))
  selected_frames = select_augmented_frames(location_df, int(num_unique_frames / 3), sigma=5)
  augmented_location_df = data_augmentation(location_df, selected_frames)

  # Combine og and augmented data
  combined_location_df = pd.concat([location_df, augmented_location_df])

  return combined_location_df


def _load_weeks(weeks, prefix, save_dir):
    tensors = [torch.load(os.path.join(save_dir, f"{prefix}_w{w}.pt")) for w in weeks]
    return torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]

@time_fcn
def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Specify constants
    all_weeks = list(range(1, 10))
    train_weeks = list(range(1, 9))
    val_weeks = [9]
    features = ["x_clean", "y_clean", "v_x", "v_y", "defense"]
    target_column = "pff_manZone"

    # Create processing dirs
    save_dir = os.path.join(os.getenv("NFL_HOME"), "data", "processed")
    os.makedirs(save_dir, exist_ok=True)

    # Load data once
    raw = RawDataLoader()
    games_df, plays_df, players_df, _ = raw.get_data(weeks=all_weeks)

    # Process each week
    for w in all_weeks:
        logging.info(f"Processing week: {w}")

        # Load just this week’s location data
        _, _, _, location_df_w = raw.get_data(weeks=[w])

        # Process (cleaning, data augmentation, etc)
        combined_loc_df_w = _process_df(location_df_w, plays_df, games_df)

        # Keep the cols that we want to reduce memory
        keep_cols = [
            "frameUniqueId", "displayName", "frameId", "frameType",
            "x_clean", "y_clean", "v_x", "v_y", "defensiveTeam",
            "pff_manZone", "defense"
        ]
        combined_loc_df_w = combined_loc_df_w[keep_cols].copy()

        # Downcast numeric columns before tensorization
        for c in ["x_clean", "y_clean", "v_x", "v_y", "defense"]:
            combined_loc_df_w[c] = pd.to_numeric(combined_loc_df_w[c], downcast="float")

        # Convert each frame to a tensor
        features_tensor_w, targets_tensor_w = prepare_frame_data(combined_loc_df_w, features=features, target_column=target_column)

        # Save per-week artifacts
        torch.save(features_tensor_w,  os.path.join(save_dir, f"features_w{w}.pt"))
        torch.save(targets_tensor_w,  os.path.join(save_dir, f"targets_w{w}.pt"))

        # Free memory for next loop
        del location_df_w, combined_loc_df_w, features_tensor_w, targets_tensor_w
        gc.collect()

    # Aggregrate train/val from saved weekly tensors
    logging.info("Aggregrating training set (Week 1 through 8)")
    train_features = _load_weeks(train_weeks, "features", save_dir)
    train_targets  = _load_weeks(train_weeks, "targets", save_dir)

    logging.info("Aggregrating validation set (Week 9)")
    val_features = _load_weeks(val_weeks, "features", save_dir)
    val_targets  = _load_weeks(val_weeks, "targets", save_dir)

    # Optional: ensure float32
    train_features = train_features.to(torch.float32)
    val_features   = val_features.to(torch.float32)

    # Save pooled artifacts
    torch.save(train_features, os.path.join(save_dir, "features_training.pt"))
    torch.save(train_targets, os.path.join(save_dir, "targets_training.pt"))
    torch.save(val_features, os.path.join(save_dir, "features_val.pt"))
    torch.save(val_targets, os.path.join(save_dir, "targets_val.pt"))

    logging.info("Train features: %s", getattr(train_features, "shape", None))
    logging.info("Val features:   %s", getattr(val_features, "shape", None))
    logging.info("Train targets:  %s", getattr(train_targets, "shape", None))
    logging.info("Val targets:    %s", getattr(val_targets, "shape", None))


if __name__ == "__main__":
  main()