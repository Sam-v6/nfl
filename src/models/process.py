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
from sklearn.model_selection import train_test_split
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

# Local imports
# None

# Load data
def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = pd.read_csv(file_path)
    return data


if __name__ == "__main__":

    #--------------------------------------------------
    # Load data
    #--------------------------------------------------
    games_df = load_data(os.path.join(os.getenv('NFL_HOME'),'data','games.csv'))
    plays_df = load_data(os.path.join(os.getenv('NFL_HOME'),'data','plays.csv'))
    players_df = load_data(os.path.join(os.getenv('NFL_HOME'),'data','players.csv'))
    week1_df = pd.read_csv(os.path.join(os.getenv('NFL_HOME'),'data','tracking_week_1.csv'))

    #--------------------------------------------------
    # Printing some summary details of the data
    #--------------------------------------------------
    print("----------------------------------------------------------------------------------------------------")
    print("Printing flavor of ingested data")
    print(games_df.head())
    print(games_df.tail())
    print(plays_df.head())
    print(players_df.head())
    print(week1_df.head())
    print("----------------------------------------------------------------------------------------------------")

    #--------------------------------------------------
    # Filter for specific game (Buffalo vs LA Rams)
    #--------------------------------------------------
    # Filter out rows with 'PENALTY' in the 'playDescription' column
    plays_df = plays_df[~plays_df['playDescription'].str.contains("PENALTY", na=False)]

    # Filter for only rows that indicate a pass play
    plays_df = plays_df[plays_df['passResult'].notna()]

    def clock_to_seconds(clock_str):
        try:
            minutes, seconds = map(int, clock_str.split(':'))
            return minutes * 60 + seconds
        except:
            return None  # handle unexpected formats

    plays_df['gameClock_seconds'] = plays_df['gameClock'].apply(clock_to_seconds)

    # Define your columns
    categorical_cols = ['offenseFormation', 'receiverAlignment', 'pff_passCoverage']
    numeric_cols = ['down', 'yardsToGo', 'quarter', 'gameClock_seconds', 'absoluteYardlineNumber']
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

    output_cols = ['yardsGained']

    #--------------------------------------------------------------------
    # Slice data into training and test
    #--------------------------------------------------------------------
    # Splitting the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(label_encoded_df, plays_df[output_cols], test_size=0.2, random_state=42)

    # Convert y_train and y_test to 1D arrays
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    print(len(x_train), "train samples")
    print(len(x_test), "test samples")

    #--------------------------------------------------------------------
    # Apply different regression approaches
    #--------------------------------------------------------------------
    # Init results dict
    results = {
        "dtr": [],           # Decision tree 
        "rfr": [],           # Random forest 
        "knn": [],           # K-nearest neighbors 
        "gbr": [],           # Gradient boosting 
        "xgbr": [],          # XGBoost 
        "catbr": [],         # CatBoost 
    }

    models = {}
    predictions = {}
    table = []
    best_transformed_features = None

    # Train different regression models
    models["dtr"] = DecisionTreeRegressor().fit(x_train, y_train)
    models["rfr"] = RandomForestRegressor().fit(x_train, y_train)
    models["knn"] = KNeighborsRegressor().fit(x_train, y_train)
    models["gbr"] = GradientBoostingRegressor().fit(x_train, y_train)
    models["xgbr"] = XGBRegressor().fit(x_train, y_train)
    models["catbr"] = CatBoostRegressor(silent=True).fit(x_train, y_train)

    # Predict and calculate rmse
    for regressor in models:
        predictions[regressor] = models[regressor].predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions[regressor]))

        # Store the results
        results[regressor].append((rmse, predictions[regressor]))

    #--------------------------------------------------------------------
    # Post process and plot different regressors for comparison
    #--------------------------------------------------------------------
    # Init table for txt output
    table = []

    # Post-process different regression approaches
    for regressor in results:
        # Extract the rmse and best predictions
        min_rmse, best_predictions = results[regressor][0]

        # Update table
        table.append([regressor, min_rmse])

        # Print the rmse
        print(f"Regressor: {regressor}")
        print(f"rmse: {min_rmse}")

        # Plotting predictions against actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, best_predictions, alpha=0.7)

        # Make sure both actual and predicted are 1D
        y_test_flat = y_test.ravel()
        preds_flat = best_predictions.ravel()

        min_val = min(min(y_test_flat), min(preds_flat))
        max_val = max(max(y_test_flat), max(preds_flat))

        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')  # Diagonal line

        plt.xlabel("Yads Gained (Actual)")
        plt.ylabel("Yards Gained (Predicted)")
        plt.title(f"Predictions vs Actual for {regressor} with rmse: {"{:.5f}".format(min_rmse)}")
        plt.savefig(os.path.join(os.getenv('NFL_HOME'), 'output', regressor + '.png'))
        plt.close()



