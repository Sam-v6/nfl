"""
Module: data.py
Description: Class for loading raw tracking data

Author: Syam Evani
Created: 2025-10-15
"""


from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.optim import AdamW
pd.options.mode.chained_assignment = None
import warnings
import random
import torch
import torch.nn as nn
import polars as pl

from models.transformer import ManZoneTransformer

def process_week_data_preds(week_number, plays):
  file_path = f"/home/sam/repos/hobby-repos/ExposingCoverageTells-BDB25/data/raw/tracking_week_{week_number}.csv"
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
  # week = week[week['frameId'] % 2 == 0]

  # ridding of any potential outliers (25 seconds after the snap)
  week = week[(week['frames_from_snap'] >= -150) & (week['frames_from_snap'] <= 30)]

  # applying data augmentation to increase training size (centered around 0-4 seconds presnap!)
  # -- 1/3rd of the current num of frames... specifically selecting for frames around the snap

  # num_unique_frames = len(set(week['frameUniqueId']))
  # selected_frames = select_augmented_frames(week, int(num_unique_frames / 3), sigma=5)
  # week_aug = data_augmentation(week, selected_frames)

  # week = pd.concat([week, week_aug])

  print(f"Finished processing Week {week_number} data")
  print()

  return week


def prepare_tensor(play, num_players=22, num_features=5):

  features = ['x_clean', 'y_clean', 'v_x', 'v_y', 'defense']
  play_data = play[features + ['frameId']]
  play_data = play_data.sort_values(by='frameId')

  frames = (
    play_data
    .groupby('frameId')[features]
    .apply(lambda g: g.to_numpy())
  )
  all_frames_tensor = np.stack(frames.to_list())  # Shape: [num_frames, num_players, num_features]
  all_frames_tensor = torch.tensor(all_frames_tensor, dtype=torch.float32)

  return all_frames_tensor  # Shape: [num_frames, num_players, num_features]


def main():


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = ManZoneTransformer(
      feature_len=5,    # num of input features (x, y, v_x, v_y, defense)
      model_dim=64,     # experimented with 96 & 128... seems best
      num_heads=2,      # 2 seems best (but may have overfit when tried 4... may be worth iterating)
      num_layers=4,
      dim_feedforward=64 * 4,
      dropout=0.1,      # 10% dropout to prevent overfitting... iterate as model becomes more complex (industry std is higher, i believe)
      output_dim=2      # man or zone classification
  ).to(device)

  # Load data
  rawLoader = RawDataLoader()
  games_df, plays_df, players_df, location_data_df = rawLoader.get_data(weeks=[i for i in range(1, 10)])

  all_weeks = []

  for week_number in range(1, 10):
    week_data = process_week_data_preds(week_number, plays_df)
    all_weeks.append(week_data)

  all_tracking = pd.concat(all_weeks, ignore_index=True)
  all_tracking = all_tracking[(all_tracking['club'] != 'football') & (all_tracking['passAttempt'] == 1)]

  # ~20mins per week
  ## -- many ways to optimize (currently isn't batched)

  for week_eval in range(1, 10):

    tracking_df = all_tracking[all_tracking['week'] == week_eval]
    tracking_df_polars = pl.DataFrame(tracking_df)  # convert to Polars

    list_ids = list(set(tracking_df['frameUniqueId']))
    save_path = os.path.join(os.getenv('NFL_HOME'), 'data', 'processed')
    best_model_path = os.path.join(save_path, f"best_model_week{week_eval}.pth")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    model.eval()

    results = []

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", category=DeprecationWarning)
        print(f"Starting loop for week {week_eval}...")

        for idx, frame_id in enumerate(list_ids, start=1):  # enumerating to print out certain invervals

            if idx % 20000 == 0:
              print(f"Processed {idx} frame_ids for week {week_eval}...")
              print(f"{idx / len(list_ids):.2f}%")

            play_id = "_".join(frame_id.split("_")[:2])
            frame_num = frame_id.split("_")[-1]
            frame_num = int(frame_num)

            frame = tracking_df_polars.filter(pl.col("frameUniqueId") == frame_id)
            frame = frame.to_pandas()

            frame_tensor = prepare_tensor(frame)
            frame_tensor = frame_tensor.to(device)  # Move to device if necessary

            with torch.no_grad():
                outputs = model(frame_tensor)  # Shape: [num_frames, num_classes]
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

                zone_prob = probabilities[0][0]
                man_prob = probabilities[0][1]

                pred = 0 if zone_prob > man_prob else 1
                actual = frame['pff_manZone'].iloc[0]

                results.append({
                    'frameUniqueId': frame_id,
                    'uniqueId': play_id,
                    'frameId': frame_num,
                    'zone_prob': zone_prob,
                    'man_prob': man_prob,
                    'pred': pred,
                    'actual': actual
                })

        week_results = pd.DataFrame(results)
        week_results.to_csv(os.path.join(save_path, f"week{week_eval}_preds.csv"))
        print(f"Finished week {week_eval}... saving to week{week_eval}_preds.csv")
        print()

  for week_eval in range(1,10):

    week_df = all_tracking[all_tracking['week'] == week_eval]
    preds_week = pd.read_csv(os.path.join(save_path, f"week{week_eval}_preds.csv"))

    preds_week = preds_week[['frameUniqueId', 'zone_prob', 'man_prob', 'pred', 'actual']]

    tracking_preds = week_df.merge(preds_week, on='frameUniqueId',how='left')

    tracking_preds.to_csv(os.path.join(save_path, f"tracking_week_{week_eval}_preds.csv"))


if __name__ == "__main__":
  main()