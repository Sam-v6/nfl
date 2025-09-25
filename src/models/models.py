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
from processing.data import DataLoader

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
    loader = DataLoader()
    games_df, plays_df, players_df, location_data_df = loader.get_data(weeks=[1, 2])

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

   