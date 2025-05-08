#/usr/bin/env python

"""
Purpose: Process nfl data for machine learning model creation
Author: Syam Evani
Date: April 2025
"""


# Standard imports
import os
import random

# General imports
import numpy as np
import itertools
import pandas as pd

# Plotting imports
import seaborn as sns
import matplotlib.pyplot as plt

# ML utils
from scipy.interpolate import griddata
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Regressors imports
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, BayesianRidge


def model_man_vs_zone(games_df, plays_df, players_df, location_data_df):
    """
    Process NFL data for machine learning model creation.
    Args:
        games_df (pd.DataFrame): DataFrame containing game data.
        plays_df (pd.DataFrame): DataFrame containing play data.
    Returns:
        None
    """

    #--------------------------------------------------
    # Data cleaning
    #--------------------------------------------------
    # Filter out rows with 'PENALTY' in the 'playDescription' column or that were nullified by penalty
    plays_df = plays_df[~plays_df['playDescription'].str.contains("PENALTY", na=False)]
    plays_df = plays_df[plays_df['playNullifiedByPenalty']=='N']

    # Filter for only rows that indicate a pass play
    #plays_df = plays_df[plays_df['passResult'].notna()]

    # Filter plays that only have man or zone coverage
    pplays_df = plays_df[plays_df['pff_manZone'].isin(['Man', 'Zone'])]

    # TODO: Remove this once downstream logic is setup, this is purely to make things go faster
    # Filter down plays_df to only plays that occur in games in week 1

    game_id_list = location_data_df['gameId'].unique()          # Grabs list of games (location_data_df is fed in only being comprised of week1)
    plays_df = plays_df[plays_df['gameId'].isin(game_id_list)]  # Filters down to only plays that have gameIds from week 1

    print(f'Total amount of samples: {len(plays_df)}')

    #--------------------------------------------------
    # Feature engineering
    #--------------------------------------------------
    #-------------------------------
    # Figure out minimum frames
    #-------------------------------
    frame_length = []
    for play_id in plays_df['playId'].unique():

        # Create a subset df of just players location data for the specific play
        game_id = plays_df[plays_df['playId'] == play_id]['gameId'].iloc[0]             # Feel like I may need gameid later?
        location_play_df = location_data_df[location_data_df['playId'] == play_id]

        # Filter down the location specific play df to pre snap
        location_play_df = location_play_df[location_play_df['frameType'] == 'BEFORE_SNAP']

        # Filter down the location specific play df to just players on defense (includes figuring out who's on defense)
        defense_team = plays_df[plays_df['playId']== play_id]['defensiveTeam'].iloc[0]
        location_play_df = location_play_df[location_play_df['club'] == defense_team]

        # Skip play if there aren't 11 defenders for some reason (Ole Billy or Vrabel filtering here)
        if location_play_df['nflId'].nunique() != 11:
            continue

        # Determine number of frames in this dataset
        example_frames_df = location_play_df[location_play_df['displayName'] == location_play_df['displayName'].iloc[0]]
        frame_count = len(example_frames_df)

        # Save frame count
        frame_length.append(frame_count)

    frame_df = pd.DataFrame(frame_length)
    min_frames = int(np.ceil(frame_df.quantile(0.20).iloc[0]))
    print(f'Minium Frames: {min_frames}')

    #-------------------------------
    # Put data into format for x input
    #-------------------------------
    X_data = []
    y_data = []
    for play_id in plays_df['playId'].unique():

        # Create a subset df of just players location data for the specific play
        game_id = plays_df[plays_df['playId'] == play_id]['gameId'].iloc[0]             # Feel like I may need gameid later?
        location_play_df = location_data_df[location_data_df['playId'] == play_id]

        # Filter down the location specific play df to pre snap
        location_play_df = location_play_df[location_play_df['frameType'] == 'BEFORE_SNAP']

        # Filter down the location specific play df to just players on defense (includes figuring out who's on defense)
        defense_team = plays_df[plays_df['playId']== play_id]['defensiveTeam'].iloc[0]
        location_play_df = location_play_df[location_play_df['club'] == defense_team]

        # Skip play if there aren't 11 defenders for some reason (Ole Billy or Vrabel filtering here)
        if location_play_df['nflId'].nunique() != 11:
            continue

        # Determine number of frames in this dataset
        example_frames_df = location_play_df[location_play_df['displayName'] == location_play_df['displayName'].iloc[0]]
        frame_count = len(example_frames_df)

        # Skip if below the number of minimum frames
        if frame_count < min_frames:
            continue

        # Squeeze data into a 2D form that will take [number of valid plays, min frames * 11 players * 2 (x, y data)]
        player_data_list = []
        for nfl_id, player_df in location_play_df.groupby('nflId'):
            player_xy = player_df.sort_values('frameId')[['x', 'y']].values # Takes shape of 2D np array as (number of frames as rows, and x and y as columns)
            player_xy = player_xy[-min_frames:]                             # Slice so we only keep the minimum number of frames (ie 80 rows to 20 rows)

            # Flatten into 1D: x1, y1, x2, y2, ..., xF, yF
            player_flat = player_xy.flatten()
            player_data_list.append(player_flat)
            
        # Concatenate all players: total length = 11 * min_frames * 2
        play_features = np.concatenate(player_data_list)
        X_data.append(play_features)

        # Set Man Coverage to 1 and Zone to 0 for y data
        # TODO: Somehow label is getting Other or Nan, need to see if it's not actually filtering and I need to do copy
        label = plays_df[plays_df['playId'] == play_id]['pff_manZone'].iloc[0]
        if label == 'Man':
            y_data.append(1)
        elif label == 'Zone':
            y_data.append(0)
        else:
            print(label)

    # Convert to arrays
    X = np.array(X_data)
    y = np.array(y_data)

    print(f"Final dataset shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y)}")  # Counts of 0 and 1

    # Validation checks
    expected_shape = min_frames * 11 * 2
    if X.shape[1] != expected_shape:
        raise ValueError(f"X.shape[1] ({X.shape[1]}) does not match expected value ({expected_shape}).")
    if X.shape[0] != np.bincount(y)[0]:
        raise ValueError("X[0] is not equal to np.bincount(y).")

    # Return
    return 0