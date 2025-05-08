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

# Local imports
from critical_down_yardage import model_critical_downs_yardage
from identify_general_coverage import model_man_vs_zone

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
    # Load all data
    games_df = load_data(os.path.join(os.getenv('NFL_HOME'),'data','games.csv'))
    plays_df = load_data(os.path.join(os.getenv('NFL_HOME'),'data','plays.csv'))
    players_df = load_data(os.path.join(os.getenv('NFL_HOME'),'data','players.csv'))
    week1_df = pd.read_csv(os.path.join(os.getenv('NFL_HOME'),'data','tracking_week_1.csv'))
    week2_df = pd.read_csv(os.path.join(os.getenv('NFL_HOME'),'data','tracking_week_2.csv'))
    week3_df = pd.read_csv(os.path.join(os.getenv('NFL_HOME'),'data','tracking_week_3.csv'))
    week4_df = pd.read_csv(os.path.join(os.getenv('NFL_HOME'),'data','tracking_week_4.csv'))
    week5_df = pd.read_csv(os.path.join(os.getenv('NFL_HOME'),'data','tracking_week_5.csv'))
    week6_df = pd.read_csv(os.path.join(os.getenv('NFL_HOME'),'data','tracking_week_6.csv'))
    week7_df = pd.read_csv(os.path.join(os.getenv('NFL_HOME'),'data','tracking_week_7.csv'))
    week8_df = pd.read_csv(os.path.join(os.getenv('NFL_HOME'),'data','tracking_week_8.csv'))
    week9_df = pd.read_csv(os.path.join(os.getenv('NFL_HOME'),'data','tracking_week_9.csv'))

    # Combine the weeks data into a single location data df
    location_data_df = pd.concat([week1_df, week2_df, week3_df, week4_df, week5_df, week6_df, week7_df, week8_df, week9_df], ignore_index=True)
    #location_data_df = week1_df


    #--------------------------------------------------
    # Printing some summary details of the data
    #--------------------------------------------------
    print("----------------------------------------------------------------------------------------------------")
    print("Printing flavor of ingested data")
    print(games_df.head())
    print(games_df.columns)
    print(plays_df.head())
    print(plays_df.columns)
    print(players_df.head())
    print(players_df.columns)
    print(location_data_df.head())
    print(location_data_df.columns)
    print("----------------------------------------------------------------------------------------------------")

    #--------------------------------------------------
    # Call models
    #--------------------------------------------------
    #model_critical_downs_yardage(games_df, plays_df)
    model_man_vs_zone(games_df, plays_df, players_df, location_data_df)

   