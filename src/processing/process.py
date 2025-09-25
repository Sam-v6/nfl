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

    #--------------------------------------------------
    # Printing some summary details of the data
    #--------------------------------------------------
    print("----------------------------------------------------------------------------------------------------")
    print("STATUS: Showing df")
    print(data.head())

    return data


# if __name__ == "__main__":

#     #--------------------------------------------------
#     # Load data
#     #--------------------------------------------------
    
#     #--------------------------------------------------
#     # Filter for specific game (Buffalo vs LA Rams)
#     #--------------------------------------------------
#     # Filter out rows with 'PENALTY' in the 'playDescription' column
#     plays_df = plays_df[~plays_df['playDescription'].str.contains("PENALTY", na=False)]

#     # Filter for only rows that indicate a pass play
#     plays_df = plays_df[plays_df['passResult'].notna()]

#     def clock_to_seconds(clock_str):
#         try:
#             minutes, seconds = map(int, clock_str.split(':'))
#             return minutes * 60 + seconds
#         except:
#             return None  # handle unexpected formats

#     plays_df['gameClock_seconds'] = plays_df['gameClock'].apply(clock_to_seconds)

#     # Define your columns
#     categorical_cols = ['offenseFormation', 'receiverAlignment', 'pff_passCoverage']
#     numeric_cols = ['down', 'yardsToGo', 'quarter', 'gameClock_seconds', 'absoluteYardlineNumber']
#     input_cols = categorical_cols + numeric_cols
