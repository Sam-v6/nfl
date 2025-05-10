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
import pandas as pd

# Plotting imports
import seaborn as sns
import matplotlib.pyplot as plt

# ML utils
from sklearn.metrics import confusion_matrix, classification_report

# Regressors imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict


def model_man_vs_zone(games_df, plays_df, players_df, location_data_df):
    """
    Process NFL data for machine learning model creation.
    Args:
        games_df (pd.DataFrame): DataFrame containing game data.
        plays_df (pd.DataFrame): DataFrame containing play data.
    Returns:
        None
    """

    random.seed(42)
    np.random.seed(42)

    #--------------------------------------------------
    # Data cleaning
    #--------------------------------------------------
    # Filter out rows
    print(f'Total plays: {len(plays_df)}')
    plays_df = plays_df[
        (~plays_df['playDescription'].str.contains("PENALTY", na=False)) &  # Filter out penalty plays
        (plays_df['playNullifiedByPenalty'] == 'N') &                       # Robustely filter out penalty plays
        (plays_df['pff_manZone'].isin(['Man', 'Zone'])) &                   # Only get plays that are man or zone
        (plays_df['gameId'].isin(location_data_df['gameId'].unique()))      # Somewhat temp, only to make things go faster and get games in week 1
    ]
    print(f'Total plays after filtering: {len(plays_df)}')

    #--------------------------------------------------
    # Feature engineering
    #--------------------------------------------------
    #-------------------------------
    # Figure out minimum frames
    #-------------------------------
    frame_length = []
    incomplete_players = []
    for i in range(0, len(plays_df)):

        # Pull the play id and game id
        game_id = plays_df.iloc[i]['gameId']
        play_id = plays_df.iloc[i]['playId']

        # Create a subset df of just players location data for the specific play
        location_play_df = location_data_df[
            (location_data_df['playId'] == play_id) &
            (location_data_df['gameId'] == game_id)
        ]

        # Filter down the location specific play df to pre snap
        location_play_df = location_play_df[location_play_df['frameType'] == 'BEFORE_SNAP']

        # Filter down the location specific play df to just players on defense (includes figuring out who's on defense)
        defense_team = plays_df.iloc[i]['defensiveTeam']
        location_play_df = location_play_df[location_play_df['club'] == defense_team]

        # Skip play if there aren't 11 defenders for some reason (Ole Billy or Vrabel filtering here)
        if location_play_df['nflId'].nunique() != 11:
            incomplete_players.append(1)
            continue

        # Determine number of frames in this dataset
        example_frames_df = location_play_df[location_play_df['displayName'] == location_play_df['displayName'].iloc[0]]
        frame_count = len(example_frames_df)

        # Save frame count
        frame_length.append(frame_count)

    frame_df = pd.DataFrame(frame_length)
    min_frames = int(np.ceil(frame_df.quantile(0.20).iloc[0]))
    print(f'Minium Frames: {min_frames}')
    print(f'Plays removed due to not 11 players: {len(incomplete_players)}')

    #-------------------------------
    # Put data into format for x input
    #-------------------------------
    X_data = []
    y_data = []
    for i in range(0, len(plays_df)):

        # Pull the play id and game id
        game_id = plays_df.iloc[i]['gameId']
        play_id = plays_df.iloc[i]['playId']

        # Create a subset df of just players location data for the specific play
        location_play_df = location_data_df[
            (location_data_df['playId'] == play_id) &
            (location_data_df['gameId'] == game_id)
        ]

        # Filter down the location specific play df to pre snap
        location_play_df = location_play_df[location_play_df['frameType'] == 'BEFORE_SNAP']

        # Filter down the location specific play df to just players on defense (includes figuring out who's on defense)
        defense_team = plays_df.iloc[i]['defensiveTeam']
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

        # Set Man Coverage to 0 and Zone to 1 for y data
        label = plays_df.iloc[i]['pff_manZone']
        if label == 'Man':
            y_data.append(0)
        elif label == 'Zone':
            y_data.append(1)
        else:
            print(label)

    # Convert to arrays
    x_array = np.array(X_data)
    y_array = np.array(y_data)

    print(f"Final dataset shape: {x_array.shape}")
    print(f"Labels distribution: {np.bincount(y_array)}")  # Counts of 0 and 1

    # Validation checks
    expected_shape = min_frames * 11 * 2
    if x_array.shape[1] != expected_shape:
        raise ValueError(f"x_array.shape[1] ({x_array.shape[1]}) does not match expected value ({expected_shape}).")
    if x_array.shape[0] != (np.bincount(y_array)[0] + np.bincount(y_array)[1]) :
        raise ValueError("x_array[0] is not equal to np.bincount(y_array).")
    
    #--------------------------------------------------
    # Build models
    #--------------------------------------------------
    # Define model
    clf = LogisticRegression(max_iter=1000)

    # Define 10-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Cross-validated accuracy scores
    scores = cross_val_score(clf, x_array, y_array, cv=skf, scoring='accuracy')
    print(f"\n10-Fold Cross-Validation Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Get cross-validated predictions to build confusion matrix
    y_pred = cross_val_predict(clf, x_array, y_array, cv=skf)

    print("\nConfusion Matrix (Cross-Validated):")
    print(confusion_matrix(y_array, y_pred))

    print("\nClassification Report (Cross-Validated):")
    print(classification_report(y_array, y_pred, target_names=['Man', 'Zone']))

    # Return
    return 0