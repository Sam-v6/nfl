# Base
import os
import random
import time
import pickle
import logging

# Common
import pandas as pd
import numpy as np

# Sklearn utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_coverage_data(games_df, plays_df, players_df, location_data_df):

    random.seed(42)
    np.random.seed(42)

    #################################
    # Filter out data we don't want
    #################################
    logging.info("Filtering data...")
    logging.info(f'Total plays: {len(plays_df)}')
    plays_df = plays_df[
        (~plays_df['playDescription'].str.contains("PENALTY", na=False)) &  # Filter out penalty plays
        (plays_df['playNullifiedByPenalty'] == 'N') &                       # Robustely filter out penalty plays
        (plays_df['pff_manZone'].isin(['Man', 'Zone'])) &                   # Only get plays that are man or zone
        (plays_df['gameId'].isin(location_data_df['gameId'].unique()))      # Somewhat temp, only to make things go faster and get games in week 1
    ]
    logging.info(f'Total plays after filtering: {len(plays_df)}')

    #################################
    # Feature engineering
    #################################


    #################################
    # Figure out minimum frames
    #################################
    min_frame_start = time.time()
    logging.info("Determining minimum frames...")
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

    logging.info(f'Minium Frames: {min_frames}')
    logging.info(f'Plays removed due to not 11 players: {len(incomplete_players)}')
    min_frame_end = time.time()
    logging.info(f"Min frame time: {min_frame_end - min_frame_start:.2f} seconds")

    #################################
    # Put data into format for x input
    #################################
    flatten_data_start = time.time()
    logging.info("Flattening data for machine learning application...")
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
            y_data.append(1)
        elif label == 'Zone':
            y_data.append(0)
        else:
            print(label)

    # Convert to arrays
    x_array = np.array(X_data)
    y_array = np.array(y_data)

    logging.info(f"Final dataset shape: {x_array.shape}")
    logging.info(f"Target distribution (Zone is 0, Man is 1): {np.bincount(y_array)}")  # Counts of 0 and 1

    # Validation checks
    expected_shape = min_frames * 11 * 2
    if x_array.shape[1] != expected_shape:
        raise ValueError(f"x_array.shape[1] ({x_array.shape[1]}) does not match expected value ({expected_shape}).")
    if x_array.shape[0] != (np.bincount(y_array)[0] + np.bincount(y_array)[1]) :
        raise ValueError("x_array[0] is not equal to np.bincount(y_array).")
    
    flatten_data_end = time.time()
    logging.info(f"Flatten data time: {flatten_data_end - flatten_data_start:.2f} seconds")
    
    #################################
    # Save data
    #################################
    # Define the directory where you want to save the files
    save_dir = os.path.join(os.getenv('NFL_HOME'), 'data', 'coverage')  # Create a directory called 'splits' in your data folder

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)  # exist_ok=True prevents errors if the directory already exists

    # Save the data splits
    with open(os.path.join(save_dir, 'x.pkl'), 'wb') as f:
        pickle.dump(x_array, f)
    with open(os.path.join(save_dir, 'y.pkl'), 'wb') as f:
        pickle.dump(y_array, f)
    logging.info(f"Data splits saved to {save_dir}")

    # Return
    return 0
