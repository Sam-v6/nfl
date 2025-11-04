#!/usr/bin/env python

"""
Module: .py
Description: Functions for loading and preprocessing test data for the thruster analysis pipeline.

Author: Syam Evani
Created: 2025-11-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os

from load_data import RawDataLoader
from clean_data import *

def process_week_data(week_number, plays):

  # -- defining function to read in all data & apply cleaning functions
  file_path = os.path.join(os.getenv("NFL_HOME"), "data", "raw", f"tracking_week_{week_number}.csv")
  week = pd.read_csv(file_path)
  print(f"Finished reading Week {week_number} data")


  # applying cleaning functions
  week = rotate_direction_and_orientation(week)
  week = make_plays_left_to_right(week)
  week = calculate_velocity_components(week)
  week = pass_attempt_merging(week, plays)

  # week = label_offense_defense_coverage(week, plays)  # for specific coverage... currently set to man/zone only
  week = label_offense_defense_manzone(week, plays)

  week['week'] = week_number
  week['uniqueId'] = week['gameId'].astype(str) + "_" + week['playId'].astype(str)
  week['frameUniqueId'] = (
      week['gameId'].astype(str) + "_" +
      week['playId'].astype(str) + "_" +
      week['frameId'].astype(str))

  # adding frames_from_snap (to do: make this a function but fine for now)
  snap_frames = week[week['frameType'] == 'SNAP'].groupby('uniqueId')['frameId'].first()
  week = week.merge(snap_frames.rename('snap_frame'), on='uniqueId', how='left')
  week['frames_from_snap'] = week['frameId'] - week['snap_frame']

  # filtering only for even frames
  week = week[week['frameId'] % 2 == 0]

  # ridding of noisier outliers out of scope (15 seconds after the snap)
  week = week[(week['frames_from_snap'] >= -150) & (week['frames_from_snap'] <= 50)]

  # applying data augmentation to increase training size (centered around 0-4 seconds presnap!)
  # -- 1/3rd of the current num of frames... specifically selecting for frames around the snap

  num_unique_frames = len(set(week['frameUniqueId']))
  selected_frames = select_augmented_frames(week, int(num_unique_frames / 3), sigma=5)
  week_aug = data_augmentation(week, selected_frames)

  week = pd.concat([week, week_aug])

  print(f"Finished processing Week {week_number} data")
  print()

  return week

def main():
  # Load data
  rawLoader = RawDataLoader()
  games_df, plays_df, players_df, location_data_df = rawLoader.get_data(weeks=[i for i in range(1, 10)])

  # Create master df
  all_weeks = []
  for week_number in range(1, 10):
    week_data = process_week_data(week_number, plays_df)
    all_weeks.append(week_data)
  all_tracking = pd.concat(all_weeks, ignore_index=True)
  all_tracking = all_tracking[(all_tracking['club'] != 'football') & (all_tracking['passAttempt'] == 1)]

  # --- takes ~10mins to run
  features = ["x_clean", "y_clean", "v_x", "v_y", "defense"]
  target_column = "pff_manZone"

  save_path = os.path.join(os.getenv('NFL_HOME'), 'data', 'processed')
  os.makedirs(save_path, exist_ok=True)
  # looping through weeks & saving each week's training data + validating data
  for week_eval in range(1, 10):

    train_df = all_tracking[all_tracking['week'] != week_eval]
    val_df = all_tracking[all_tracking['week'] == week_eval]

    train_df = train_df[['frameUniqueId', 'displayName', 'frameId', 'frameType', 'x_clean', 'y_clean', 'v_x', 'v_y', 'defensiveTeam', 'pff_manZone','defense']]
    val_df = val_df[['frameUniqueId', 'displayName', 'frameId', 'frameType', 'x_clean', 'y_clean', 'v_x', 'v_y', 'defensiveTeam', 'pff_manZone', 'defense']]

    train_features, train_targets = prepare_frame_data(train_df, features, target_column)
    val_features, val_targets = prepare_frame_data(val_df, features, target_column)

    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)

    print(f"Week {week_eval} Tensor: {train_features.shape}")       # should be: torch.Size([X, 22, 5]) where X is num frames, 22 is num plays, 5 is num features (as definied above)
    print(f"Week {week_eval} Indiv Check: {train_features[63][0]}") # should be: tensor([x_cord, y_cord, v_x,  v_y,  1/0])

    # prints the same train_features[63][0] which doesn't give much context to potential errors... to do: change to random
    torch.save(train_features, os.path.join(save_path, f"features_training_week{week_eval}preds.pt"))
    torch.save(train_targets, os.path.join(save_path, f"targets_training_week{week_eval}preds.pt"))

    torch.save(val_features, os.path.join(save_path, f"features_val_week{week_eval}preds.pt"))
    torch.save(val_targets, os.path.join(save_path, f"targets_val_week{week_eval}preds.pt"))

if __name__ == "__main__":
  main()