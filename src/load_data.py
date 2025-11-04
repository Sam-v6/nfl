#!/usr/bin/env python

"""
Module: data.py
Description: Class for loading raw tracking data

Author: Syam Evani
Created: 2025-10-15
"""

import os
import pandas as pd
from typing import Tuple

class RawDataLoader:
    def __init__(self, nfl_home=None):
        self.data_path = os.path.join(os.getenv('NFL_HOME'), 'data', 'parquet')
        self.games_df = None
        self.plays_df = None
        self.players_df = None
        self.location_data_df = None

    def _load_parquet(self, filepath):
        """Load parquet data with basic error handling"""
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def _load_base_data(self):
        """Load games, plays, and players data"""
        self.games_df = self._load_parquet(os.path.join(self.data_path, 'games.parquet'))
        self.plays_df = self._load_parquet(os.path.join(self.data_path, 'plays.parquet'))
        self.players_df = self._load_parquet(os.path.join(self.data_path, 'players.parquet'))

    def _load_tracking_data(self, weeks=list[int]):
        """Load tracking data for specified weeks"""
        weekly_dfs = []
        
        for week in weeks:
            filepath = os.path.join(self.data_path, f'tracking_week_{week}.parquet')
            df = self._load_parquet(filepath)
            if df is not None:
                weekly_dfs.append(df)
            else:
                print(f"Skipping week {week} due to load error.")
        
        self.location_data_df = pd.concat(weekly_dfs, ignore_index=True)

    def get_data(self, weeks: list[int] = range(1, 10)) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return the processed data"""

        # Load data
        self._load_base_data()
        self._load_tracking_data(weeks=weeks)

        return (self.games_df, self.plays_df, self.players_df, self.location_data_df)

def main():
    # Example usage so that I can selectively load in a few weeks to keep mem down
    loader = RawDataLoader()
    games_df, plays_df, players_df, location_data_df = loader.get_data(weeks=[1, 2])

if __name__ == "__main__":
    main()

