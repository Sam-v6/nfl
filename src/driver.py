#/usr/bin/env python

"""
Purpose: Process nfl data for machine learning model creation
Author: Syam Evani
Date: April 2025
"""


# Standard imports
import os
import random

# Additonal imports


# Local imports
from data_processing.process import load_data
from data_processing.draw_play import animate_play


if __name__ == "__main__":

    # Set random seed for reproducibility
    random.seed(42)

    # Load data
    games_df = load_data(os.path.join(os.getenv('NFL_HOME'),'data','games.csv'))
    plays_df = load_data(os.path.join(os.getenv('NFL_HOME'),'data','plays.csv'))
    players_df = load_data(os.path.join(os.getenv('NFL_HOME'),'data','players.csv'))
    week1_df = load_data(os.path.join(os.getenv('NFL_HOME'),'data','tracking_week_1.csv'))

    # Draw play frame
    animate_play(week1_df, 2022091200, 64)