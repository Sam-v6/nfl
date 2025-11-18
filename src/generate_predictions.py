#!/usr/bin/env python

"""
Trains LSTM model on location tracking data

Requires:
- Raw location tracking data in nfl/data/parquet
- Trained model in nfl/data/processed/model.pth

Performs inference on each week and saves to nfl/data/processed/week{n}_predictions.csv
"""

import math
import warnings
import random
import os
import logging

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

from models.transformer import ManZoneTransformer
from load_data import RawDataLoader
from clean_data import *
from common.paths import PROJECT_ROOT, SAVE_DIR

def process_week_data_preds(week_number, plays):
  WEEK_PARQUET_PATH = PROJECT_ROOT /  "data" / "parquet" / f"tracking_week_{week_number}.parquet"
  week = pd.read_parquet(WEEK_PARQUET_PATH)
  logging.info(f"Finished reading Week {week_number} data")

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

  logging.info(f"Finished processing Week {week_number} data")

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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    model.load_state_dict(torch.load(SAVE_DIR / "model.pth", map_location=device))
    model.eval()

    # Load data
    rawLoader = RawDataLoader()
    games_df, plays_df, players_df, location_data_df = rawLoader.get_data(weeks=[i for i in range(1, 10)])

    # Process + predict one week at a time (keeps RAM low)
    for week_eval in [9]:
        week_df = process_week_data_preds(week_eval, plays_df)

        # filter early to shrink memory
        week_df = week_df[(week_df['club'] != 'football') & (week_df['passAttempt'] == 1)].copy()

        # Polars convert optional; if you like the speed, keep it—otherwise skip to use pandas only
        tracking_df_polars = pl.DataFrame(week_df)  # or comment this out and use pandas below

        # Stream predictions to CSV in batches
        weekly_predictions_path_csv = SAVE_DIR / f"week{week_eval}_preds.csv"
        wrote_header = False
        batch = []
        BATCH_SIZE = 5000  # tune (smaller => lower peak RAM)

        # Iterate unique frames without building a giant Python set
        list_ids = pd.unique(week_df['frameUniqueId'].values)

        logging.info(f"Starting loop for week {week_eval}...")
        for idx, frame_id in enumerate(list_ids, start=1):
            if idx % 20000 == 0:
                logging.info(f"Processed {idx}/{len(list_ids)} frames ({100*idx/len(list_ids):.1f}%)")

            # Grab frame rows (polars or pandas)
            if tracking_df_polars is not None:
                frame = tracking_df_polars.filter(pl.col("frameUniqueId") == frame_id).to_pandas()
            else:
                frame = week_df.loc[week_df["frameUniqueId"] == frame_id]

            # Lightweight tensor build
            frame_tensor = prepare_tensor(frame)
            if frame_tensor is None:
                continue

            frame_tensor = frame_tensor.to(device, non_blocking=True)

            with torch.no_grad():
                outputs = model(frame_tensor)                 # [1, 2]
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                zone_prob, man_prob = float(probabilities[0]), float(probabilities[1])
                pred = 0 if zone_prob > man_prob else 1
                actual = int(frame['pff_manZone'].iloc[0]) if 'pff_manZone' in frame.columns and not pd.isna(frame['pff_manZone'].iloc[0]) else -1

            play_id = "_".join(frame_id.split("_")[:2])
            frame_num = int(frame_id.split("_")[-1])

            batch.append({
                'frameUniqueId': frame_id,
                'uniqueId': play_id,
                'frameId': frame_num,
                'zone_prob': zone_prob,
                'man_prob': man_prob,
                'pred': pred,
                'actual': actual
            })

            # Flush batch to CSV to keep RAM low
            if len(batch) >= BATCH_SIZE:
                pd.DataFrame(batch).to_csv(weekly_predictions_path_csv, mode='a', header=not wrote_header, index=False)
                wrote_header = True
                batch.clear()

        # Flush tail
        if batch:
            pd.DataFrame(batch).to_csv(weekly_predictions_path_csv, mode='a', header=not wrote_header, index=False)
            batch.clear()

        logging.info(f"Finished week {week_eval}... saved to week{week_eval}_preds.csv\n")

        # Merge week_df with preds (per-week, small)
        preds_week = pd.read_csv(weekly_predictions_path_csv, usecols=['frameUniqueId','zone_prob','man_prob','pred', 'actual'])
        tracking_preds = week_df.merge(preds_week, on='frameUniqueId', how='left')
        tracking_preds.to_csv(SAVE_DIR / f"tracking_week_{week_eval}_preds.csv", index=False)

        # Free RAM before next week
        del week_df, tracking_df_polars, preds_week, tracking_preds
        import gc; gc.collect()

if __name__ == "__main__":
  main()