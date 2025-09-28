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


def model_critical_downs_yardage(games_df, plays_df):
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

    # Filter out rows with 'PENALTY' in the 'playDescription' column
    plays_df = plays_df[~plays_df['playDescription'].str.contains("PENALTY", na=False)]

    # Filter for only rows that indicate a pass play
    plays_df = plays_df[plays_df['passResult'].notna()]

    # Filter for only plays where the win probablity isn't lopsided (between 0.2 and 0.8)
    plays_df = plays_df[(plays_df['preSnapHomeTeamWinProbability'] > 0.2) & (plays_df['preSnapHomeTeamWinProbability'] < 0.8)]

    # Filter for only third down or fourth down plays
    plays_df = plays_df[plays_df['down'].isin([3, 4])]

    # Filter for only 3rd/4th and long (equal or more than 5 yards to go)
    plays_df = plays_df[plays_df['yardsToGo'] >= 5]

    print(f'Total amount of samples: {len(plays_df)}')

    #--------------------------------------------------
    # Feature engineering
    #--------------------------------------------------
    # Create new feature - gameClockSeconds (convert game clock to seconds)
    def clock_to_seconds(clock_str):
        try:
            minutes, seconds = map(int, clock_str.split(':'))
            return minutes * 60 + seconds
        except:
            return None  # handle unexpected formats

    plays_df['gameClockSeconds'] = plays_df['gameClock'].apply(clock_to_seconds)

    # Create new feature - offenseScoreDelta
    def calculate_offense_score_delta(row):
        # Get the home team abbreviation for the current game_id
        home_team_abbr = games_df.loc[games_df['gameId'] == row['gameId'], 'homeTeamAbbr'].values[0]
        
        # Calculate the offense score delta based on possession team
        if row['possessionTeam'] == home_team_abbr:
            return row['preSnapHomeScore'] - row['preSnapVisitorScore']
        else:
            return row['preSnapVisitorScore'] - row['preSnapHomeScore']

    plays_df['offenseScoreDelta'] = plays_df.apply(calculate_offense_score_delta, axis=1)
    print(plays_df['offenseScoreDelta'].describe())

    # Defining columns
    categorical_cols = ['offenseFormation', 'receiverAlignment', 'pff_passCoverage']
    numeric_cols = ['down', 'yardsToGo', 'quarter', 'gameClockSeconds', 'absoluteYardlineNumber', 'offenseScoreDelta']
    input_cols = categorical_cols + numeric_cols

    # Make a copy of just those columns
    input_df = plays_df[input_cols].copy()

    # Label encode categorical columns
    label_encoded_df = input_df.copy()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        label_encoded_df[col] = le.fit_transform(label_encoded_df[col].astype(str))
        label_encoders[col] = le  # Save encoder

    # Scale numeric columns
    scaler = StandardScaler()
    label_encoded_df[numeric_cols] = scaler.fit_transform(label_encoded_df[numeric_cols])

    #--------------------------------------------------
    # Target engineering
    #--------------------------------------------------
    output_cols = ['yardsGained']

    corr_columns = numeric_cols + ['yardsGained']
    corr_matrix = plays_df[corr_columns].corr()
    print(corr_matrix["yardsGained"].sort_values(ascending=False))

    #--------------------------------------------------
    # Step 1: Split into training and test sets
    #--------------------------------------------------
    X = label_encoded_df
    y = plays_df[output_cols].values.ravel()  # Flatten output

    # Hold out 20% as the final test set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #--------------------------------------------------
    # Step 2: Perform 10-fold cross-validation on training data
    #--------------------------------------------------
    # Setup the models that will be used for k-fold cross-validation (trying to see what performs the best)
    models = {
        "dtr": DecisionTreeRegressor(),             # Decision tree 
        "rfr": RandomForestRegressor(),             # Random forest 
        "knn": KNeighborsRegressor(),               # K-nearest neighbors 
        "gbr": GradientBoostingRegressor(),         # Gradient boosting 
        "xgbr": XGBRegressor(),                     # XGBoost 
        "catbr": CatBoostRegressor(silent=True),    # CatBoost 
    }

    # Set up folds
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_rmse = {key: [] for key in models.keys()}  # Store RMSE for each fold

    # Perform k-fold cross-validation
    # Note: this is picking random indexes each iteration of data to call training or val: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    for fold, (train_index, val_index) in enumerate(kf.split(x_train)):                 
        x_fold_train, x_fold_val = x_train.iloc[train_index], x_train.iloc[val_index]   
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

        # Train each model on the training fold
        for key in models:

            # Fit model
            models[key].fit(x_fold_train, y_fold_train)

            # Predict on validation fold
            y_pred = models[key].predict(x_fold_val)

            # Calculate, store, and print RMSE
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            fold_rmse[key].append(rmse)
            print(f"Fold {fold+1}, Model: {key}, MSE: {rmse:.4f}")

    # Calculate average RMSE for each model across folds
    for key, values in fold_rmse.items():
        fold_rmse[key] = np.mean(values)
        print(f"Model: {key}, Average CV RMSE: {fold_rmse[key]:.4f}")

