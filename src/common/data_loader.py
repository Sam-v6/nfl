import os
import pandas as pd

# Typing
from typing import Tuple


class DataLoader:
    def __init__(self, nfl_home=None):
        self.data_path = os.path.join(os.getenv('NFL_HOME'), 'data', 'raw')
        self.games_df = None
        self.plays_df = None
        self.players_df = None
        self.location_data_df = None

    def _load_data(self, filepath):
        """Load CSV data with basic error handling"""
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def _load_base_data(self):
        """Load games, plays, and players data"""
        self.games_df = self._load_data(os.path.join(self.data_path, 'games.csv'))
        self.plays_df = self._load_data(os.path.join(self.data_path, 'plays.csv'))
        self.players_df = self._load_data(os.path.join(self.data_path, 'players.csv'))

    def _load_tracking_data(self, weeks=range(1, 10)):
        """Load tracking data for specified weeks"""
        weekly_dfs = []
        
        for week in weeks:
            filepath = os.path.join(self.data_path, f'tracking_week_{week}.csv')
            df = self._load_data(filepath)
            if df is not None:
                weekly_dfs.append(df)
        
        self.location_data_df = pd.concat(weekly_dfs, ignore_index=True)

    def _clean_data(self):
        """Placeholder for data cleaning operations"""
        pass

    def _process_data(self):
        """Placeholder for data processing operations"""
        pass

    def get_data(self, weeks: list[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return the processed data"""

        self._load_base_data()
        self._load_tracking_data(weeks=weeks)
        self._clean_data()
        self._process_data()

        return (self.games_df, self.plays_df, self.players_df, self.location_data_df)


if __name__ == "__main__":

    # Example usage so that I can selectively load in a few weeks to keep mem down
    loader = DataLoader()
    games_df, plays_df, players_df, location_data_df = loader.get_data(weeks=[1, 2])