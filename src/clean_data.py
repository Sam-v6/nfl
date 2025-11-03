"""
Module: data_loader.py
Description: Functions for loading and preprocessing test data for the thruster analysis pipeline.

Author: Syam Evani
Created: 2025-11-02
"""

import numpy as np
import pandas as pd
import math
import torch

def rotate_direction_and_orientation(df):

  """
  Rotate the direction and orientation angles so that 0° points from left to right on the field, and increasing angle goes counterclockwise
  This should be done BEFORE the call to make_plays_left_to_right, because that function with compensate for the flipped angles.

  :param df: the aggregate dataframe created using the aggregate_data() method

  :return df: the aggregate dataframe with orientation and direction angles rotated 90° clockwise
  """

  df["o_clean"] = (-(df["o"] - 90)) % 360
  df["dir_clean"] = (-(df["dir"] - 90)) % 360

  return df


def make_plays_left_to_right(df):

  """
  Flip tracking data so that all plays run from left to right. The new x, y, s, a, dis, o, and dir data
  will be stored in new columns with the suffix "_clean" even if the variables do not change from their original value.

  :param df: the aggregate dataframe created using the aggregate_data() method

  :return df: the aggregate dataframe with the new columns such that all plays run left to right
  """

  df["x_clean"] = np.where(
      df["playDirection"] == "left",
      120 - df["x"],
      df[
          "x"
      ],  # 120 because the endzones (10 yds each) are included in the ["x"] values
  )

  df["y_clean"] = df["y"]
  df["s_clean"] = df["s"]
  df["a_clean"] = df["a"]
  df["dis_clean"] = df["dis"]

  df["o_clean"] = np.where(
      df["playDirection"] == "left", 180 - df["o_clean"], df["o_clean"]
  )

  df["o_clean"] = (df["o_clean"] + 360) % 360  # remove negative angles

  df["dir_clean"] = np.where(
      df["playDirection"] == "left", 180 - df["dir_clean"], df["dir_clean"]
  )

  df["dir_clean"] = (df["dir_clean"] + 360) % 360  # remove negative angles

  return df


def calculate_velocity_components(df):
    """
    Calculate the velocity components (v_x and v_y) for each row in the dataframe.

    :param df: the aggregate dataframe with "_clean" columns created using make_plays_left_to_right()

    :return df: the dataframe with additional columns 'v_x' and 'v_y' representing the velocity components
    """

    df["dir_radians"] = np.radians(df["dir_clean"])

    df["v_x"] = df["s_clean"] * np.cos(df["dir_radians"])
    df["v_y"] = df["s_clean"] * np.sin(df["dir_radians"])


    return df


def label_offense_defense_coverage(presnap_df, plays_df):

  coverage_replacements = {
    'Cover-3 Cloud Right': 'Cover-3',
    'Cover-3 Cloud Left': 'Cover-3',
    'Cover-3 Seam': 'Cover-3',
    'Cover-3 Double Cloud': 'Cover-3',
    'Cover-6 Right': 'Cover-6',
    'Cover 6-Left': 'Cover-6',
    'Cover-1 Double': 'Cover-1'}

  values_to_drop = ["Miscellaneous", "Bracket", "Prevent", "Red Zone", "Goal Line"]

  plays_df['pff_passCoverage'] = plays_df['pff_passCoverage'].replace(coverage_replacements)

  plays_df = plays_df.dropna(subset=['pff_passCoverage'])
  plays_df = plays_df[~plays_df['pff_passCoverage'].isin(values_to_drop)]

  coverage_mapping = {
      'Cover-0': 0,
      'Cover-1': 1,
      'Cover-2': 2,
      'Cover-3': 3,
      'Quarters': 4,
      '2-Man': 5,
      'Cover-6': 6
  }

  merged_df = presnap_df.merge(
      plays_df[['gameId', 'playId', 'possessionTeam', 'defensiveTeam', 'pff_passCoverage']],
      on=['gameId', 'playId'],
      how='left'
  )

  merged_df['defense'] = ((merged_df['club'] == merged_df['defensiveTeam']) & (merged_df['club'] != 'football')).astype(int)

  merged_df['pff_passCoverage'] = merged_df['pff_passCoverage'].map(coverage_mapping)
  merged_df.dropna(subset=['pff_passCoverage'], inplace=True)

  return merged_df


def label_offense_defense_manzone(presnap_df, plays_df):

  plays_df = plays_df.dropna(subset=['pff_manZone'])

  coverage_mapping = {
      'Zone': 0,
      'Man': 1}

  merged_df = presnap_df.merge(
      plays_df[['gameId', 'playId', 'possessionTeam', 'defensiveTeam', 'pff_manZone']],
      on=['gameId', 'playId'],
      how='left'
  )

  merged_df['defense'] = ((merged_df['club'] == merged_df['defensiveTeam']) & (merged_df['club'] != 'football')).astype(int)

  merged_df['pff_manZone'] = merged_df['pff_manZone'].map(coverage_mapping)
  merged_df.dropna(subset=['pff_manZone'], inplace=True)

  return merged_df


def label_offense_defense_formation(presnap_df, plays_df):

  """
  Adds 'offense' and 'defense' columns to presnap_df, marking players as offense (1) or defense (0)
  based on possession team and defensive team from plays_df. Enumerates offensive formations
  and removes rows with missing formations.

  Parameters:
  presnap_df (pd.DataFrame): DataFrame containing tracking data with 'gameId', 'playId', and 'club'.
  plays_df (pd.DataFrame): DataFrame containing 'gameId', 'playId', 'possessionTeam', 'defensiveTeam', 'offenseFormation'.

  Returns:
  pd.DataFrame: Updated presnap_df with added 'offense', 'defense', and enumerated 'offenseFormation' columns, with NaN formations dropped.
  """

  formation_mapping = {
      'EMPTY': 0,
      'I_FORM': 1,
      'JUMBO': 2,
      'PISTOL': 3,
      'SHOTGUN': 4,
      'SINGLEBACK': 5,
      'WILDCAT': 6
  }

  merged_df = presnap_df.merge(
      plays_df[['gameId', 'playId', 'possessionTeam', 'defensiveTeam', 'offenseFormation']],
      on=['gameId', 'playId'],
      how='left'
  )

  merged_df['defense'] = ((merged_df['club'] == merged_df['defensiveTeam']) & (merged_df['club'] != 'football')).astype(int)

  merged_df['offenseFormation'] = merged_df['offenseFormation'].map(formation_mapping)
  merged_df.dropna(subset=['offenseFormation'], inplace=True)

  return merged_df


def split_data_by_uniqueId(df, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, unique_id_column="uniqueId"):

  """
  Split the dataframe into training, testing, and validation sets based on a given ratio while
  ensuring all rows with the same uniqueId are in the same set.

  :param df: the aggregate dataframe containing all frames for each play
  :param train_ratio: proportion of the data to allocate to training (default 0.7)
  :param test_ratio: proportion of the data to allocate to testing (default 0.15)
  :param val_ratio: proportion of the data to allocate to validation (default 0.15)
  :param unique_id_column: the name of the column containing the unique identifiers for each play

  :return: three dataframes (train_df, test_df, val_df) for training, testing, and validation
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


def pass_attempt_merging(tracking, plays):

  plays['passAttempt'] = np.where(plays['passResult'].isin([np.nan, 'S']), 0, 1)

  plays_for_merge = plays[['gameId', 'playId', 'passAttempt']]

  merged_df = tracking.merge(
      plays_for_merge,
      on=['gameId', 'playId'],
      how='left')

  return merged_df


def prepare_frame_data(df, features, target_column):

  features_array = df.groupby("frameUniqueId")[features].apply(
      lambda x: x.to_numpy(dtype=np.float32)).to_numpy()

  try:
      features_tensor = torch.tensor(np.stack(features_array))
  except ValueError as e:
      print("Skipping batch due to inconsistent shapes in features_array:", e)
      return None, None  # or return some placeholder values if needed

  targets_array = df.groupby("frameUniqueId")[target_column].first().to_numpy()
  targets_tensor = torch.tensor(targets_array, dtype=torch.long)

  return features_tensor, targets_tensor


def select_augmented_frames(df, num_samples, sigma=5):

    df_frames = df[['frameUniqueId', 'frames_from_snap']].drop_duplicates()
    weights = np.exp(-((df_frames['frames_from_snap'] + 10) ** 2) / (2 * sigma ** 2))

    weights /= weights.sum()

    selected_frames = np.random.choice(
        df_frames['frameUniqueId'], size=num_samples, replace=False, p=weights
    )

    return selected_frames


def data_augmentation(df, augmented_frames):

  df_sample = df.loc[df['frameUniqueId'].isin(augmented_frames)].copy()

  df_sample['y_clean'] = (160 / 3) - df_sample['y_clean']
  df_sample['dir_radians'] = (2 * np.pi) - df_sample['dir_radians']
  df_sample['dir_clean'] = np.degrees(df_sample['dir_radians'])

  df_sample['frameUniqueId'] = df_sample['frameUniqueId'].astype(str) + '_aug'

  return df_sample
