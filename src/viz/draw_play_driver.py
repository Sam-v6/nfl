#!/usr/bin/env python

"""
Driver code to animate play for all given plays
"""

import os
import random

from data_processing.process import load_data
from data_processing.draw_play import animate_play

from common.paths import PROJECT_ROOT


if __name__ == "__main__":

    # Set random seed for reproducibility
    random.seed(42)

    # Load data
    games_df = load_data(PROJECT_ROOT / 'data' / 'raw' / 'games.csv')
    plays_df = load_data(PROJECT_ROOT / 'data' / 'raw' /  'plays.csv')
    players_df = load_data(PROJECT_ROOT / 'data' / 'raw' /  'players.csv')
    week1_df = load_data(PROJECT_ROOT / 'data' / 'raw' /  'tracking_week_1.csv')

    # Draw play frame
    for i in range(0,len(games_df)):
        # Get the game df and id
        game_info_df = games_df.iloc[i]
        game_id = game_info_df['gameId']

        # Filter the plays to only the current game
        single_game_plays_df = plays_df[plays_df['gameId'] == game_id]

        # Loop across the plays in the game
        for play_id in single_game_plays_df['playId'].unique():

            # Filter the tracking data to only the current play
            location_data_single_play_df = week1_df[(week1_df['gameId'] == game_id) & (week1_df['playId'] == play_id)]

            # Filter all the play data to the current play
            play_data_single_play_df = single_game_plays_df[single_game_plays_df['playId'] == play_id]

            if not location_data_single_play_df.empty:
                animate_play(game_info_df, play_data_single_play_df, players_df, location_data_single_play_df)
            else:
                print(f"No tracking data found for game {game_id}, play {play_id}.")