#!/usr/bin/env python

"""
Loads raw parquet data for usage in training models
"""

import os
import pandas as pd
from typing import Tuple

from common.paths import PROJECT_ROOT


class RawDataLoader:
    def __init__(self):
        if os.getenv("CI_DATA_ROOT"):
            self.DATA_PATH = os.getenv("CI_DATA_ROOT")
        else:
            self.DATA_PATH = PROJECT_ROOT / "data" / "parquet"
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
        self.games_df = self._load_parquet(self.DATA_PATH / 'games.parquet')
        self.plays_df = self._load_parquet(self.DATA_PATH / 'plays.parquet')
        self.players_df = self._load_parquet(self.DATA_PATH / 'players.parquet')

    def _load_tracking_data(self, weeks=list[int]):
        """Load tracking data for specified weeks"""
        weekly_dfs = []
        
        for week in weeks:
            filepath = self.DATA_PATH / f'tracking_week_{week}.parquet'
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

