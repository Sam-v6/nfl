#/usr/bin/env python

"""
Purpose: Process nfl data for machine learning model creation
Author: Syam Evani
Date: April 2025
"""


# Standard imports
import os
import random
import time
import pickle

# General imports
import numpy as np
import pandas as pd

# Plotting imports
import seaborn as sns
import matplotlib.pyplot as plt

# ML utils
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score, f1_score, classification_report, roc_curve, RocCurveDisplay
from sklearn.preprocessing import StandardScaler

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


def process_data(games_df, plays_df, players_df, location_data_df):

    random.seed(42)
    np.random.seed(42)

    #--------------------------------------------------
    # Filter out data we don't want
    #--------------------------------------------------
    print("----------------------------------------------------------")
    print("STATUS: Filtering data...")
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
    min_frame_start = time.time()
    print("----------------------------------------------------------")
    print("STATUS: Determining minimum frames...")
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
    min_frame_end = time.time()
    print(f"Min frame time: {min_frame_end - min_frame_start:.2f} seconds")

    #-------------------------------
    # Put data into format for x input
    #-------------------------------
    flatten_data_start = time.time()
    print("----------------------------------------------------------")
    print("STATUS: Flattening data for machine learning application...")
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

    print(f"Final dataset shape: {x_array.shape}")
    print(f"Target distribution (Zone is 0, Man is 1): {np.bincount(y_array)}")  # Counts of 0 and 1

    # Validation checks
    expected_shape = min_frames * 11 * 2
    if x_array.shape[1] != expected_shape:
        raise ValueError(f"x_array.shape[1] ({x_array.shape[1]}) does not match expected value ({expected_shape}).")
    if x_array.shape[0] != (np.bincount(y_array)[0] + np.bincount(y_array)[1]) :
        raise ValueError("x_array[0] is not equal to np.bincount(y_array).")
    
    flatten_data_end = time.time()
    print(f"Flatten data time: {flatten_data_end - flatten_data_start:.2f} seconds")
    
    #-------------------------------
    # Scale data and save it off
    #-------------------------------
    print("----------------------------------------------------------")
    print("STATUS: Scaling and saving training and test data...")
    scaler = StandardScaler()
    x_array_scaled = scaler.fit_transform(x_array)
    x_train, x_test, y_train, y_test = train_test_split(x_array_scaled, y_array, test_size=0.2, stratify=y_array, random_state=42)

    # Define the directory where you want to save the files
    save_dir = os.path.join(os.getenv('NFL_HOME'), 'data', 'coverage')  # Create a directory called 'splits' in your data folder

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)  # exist_ok=True prevents errors if the directory already exists

    # Save the data splits
    with open(os.path.join(save_dir, 'x_train.pkl'), 'wb') as f:
        pickle.dump(x_train, f)
    with open(os.path.join(save_dir, 'x_test.pkl'), 'wb') as f:
        pickle.dump(x_test, f)
    with open(os.path.join(save_dir, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(save_dir, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    print(f"Data splits saved to {save_dir}")

    # Return
    return 0

def create_models(x_train, y_train, x_test, y_test):

    #--------------------------------------------------
    # Build models
    #--------------------------------------------------
    # NOTES
    # - Consider oversampling/undersampling techniues like SMOTE, RandomOverSampler
    # - Organize the data into positions and then analyze feature importance
    # - Grid search on parameters
    # - Try additional tree based models XGBoost and LightGBM
    # - Save data out that I can load it later

    model_data_start = time.time()
    print("----------------------------------------------------------")
    print("STATUS: Bulding models...")

    models = {}
    models['log'] = LogisticRegression(max_iter=1000, class_weight='balanced')
    models['rfr'] = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    models['xgb'] = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
    #models['knn'] = KNeighborsClassifier(n_neighbors=5)
    #models['svm'] = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    models['lgb'] = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    # Define 10-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for model_name, model in models.items():
        print("----------------------------------------------------------")
        print(f"Model: {model_name}")

        if model_name == 'lgb':
            x_test = pd.DataFrame(x_test)
            x_train = pd.DataFrame(x_train)
            y_train = pd.Series(y_train).ravel()
            y_test = pd.Series(y_test).ravel()

        # K Fold validation
        print("K-Fold Cross-Validation on Training Data:")
        cv_scores = cross_val_score(model, x_train, y_train, cv=skf, scoring='roc_auc')
        print(f"10-Fold Cross-Validation ROC AUC Scores: {cv_scores}")
        print(f"Mean ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Fit on training data
        model.fit(x_train, y_train)

        # Predict on test data
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)

        # Metrics on Test Data
        print(f"Precision (Man): {precision_score(y_test, y_pred, pos_label=1):.4f}")
        print(f"Recall (Man): {recall_score(y_test, y_pred, pos_label=1):.4f}")
        print(f"F1 Score (Man): {f1_score(y_test, y_pred, pos_label=1):.4f}")
        print(f"Log Loss: {log_loss(y_test, y_proba):.4f}")
        print(f"Overall ROC AUC: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")
        print(classification_report(y_test, y_pred, target_names=['Zone', 'Man']))

        # Plot ROC AUC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1], pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba[:, 1]):.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name.upper()}')
        plt.legend(loc='lower right')
        plt.grid(True)
        image_name = f'man_zone_{model_name}_roc_auc.png'
        plt.savefig(os.path.join(os.getenv('NFL_HOME'), 'output', 'coverage', image_name))

    model_data_end = time.time()
    print(f"Model generation time: {model_data_end - model_data_start:.2f} seconds")

    # Return 
    return 0

def load_data():
    RUN_DATA_PROCESSING = False
    base_path = os.path.join(os.getenv('NFL_HOME'), 'data', 'coverage')
    data_file_list = ['x_train', 'y_train', 'x_test', 'y_test']
    data_dict = {}
    for file in data_file_list:
        file_name = file + '.pkl'
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            print(f"File {file_name} does not exist. Exiting...")
            RUN_DATA_PROCESSING = True
        else:
            with open(file_path, 'rb') as f:
                data_dict[file] = pickle.load(f)

    return RUN_DATA_PROCESSING, data_dict
    
def model_man_vs_zone(games_df, plays_df, players_df, location_data_df): 

    # Check if we should run the data processing pipeline
    RUN_DATA_PROCESSING, data_dict = load_data()
    
    # Run data processing if neccesary
    if RUN_DATA_PROCESSING:
        process_data(games_df, plays_df, players_df, location_data_df)
        _, data_dict = load_data()

    # Create models
    create_models(data_dict['x_train'], data_dict['y_train'], data_dict['x_test'], data_dict['y_test'])

    # Return
    return 0