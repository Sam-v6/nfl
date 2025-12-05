#!/usr/bin/env python
"""
Loads raw parquet data into pandas DataFrames for modeling.
"""

import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from common.paths import PROJECT_ROOT


class RawDataLoader:
	"""
	Convenience loader for raw games, plays, players, and tracking parquet files.
	"""

	def __init__(self) -> None:
		"""
		Builds a data loader for pulling raw parquet files.

		Inputs:
		- None

		Outputs:
		- Initialized data loader.
		"""

		if os.getenv("CI_DATA_ROOT"):
			self.DATA_PATH = Path(os.getenv("CI_DATA_ROOT"))
		else:
			self.DATA_PATH = PROJECT_ROOT / "data" / "parquet"
		self.games_df = None
		self.plays_df = None
		self.players_df = None
		self.location_data_df = None

	def _load_parquet(self, filepath: Path) -> pd.DataFrame | None:
		"""
		Loads a parquet file with basic error handling.

		Inputs:
		- filepath: Location of the parquet file.

		Outputs:
		- df_or_none: DataFrame on success, None on failure.
		"""

		try:
			return pd.read_parquet(filepath)
		except Exception as e:
			print(f"Error loading {filepath}: {e}")
			return None

	def _load_base_data(self) -> None:
		"""
		Pulls games, plays, and players tables into memory.

		Inputs:
		- None.

		Outputs:
		- Populates internal DataFrame attributes.
		"""

		self.games_df = self._load_parquet(self.DATA_PATH / "games.parquet")
		self.plays_df = self._load_parquet(self.DATA_PATH / "plays.parquet")
		self.players_df = self._load_parquet(self.DATA_PATH / "players.parquet")

	def _load_tracking_data(self, weeks: Iterable[int]) -> None:
		"""
		Loads weekly tracking parquet files and concatenates them.

		Inputs:
		- weeks: Iterable of week numbers to include.

		Outputs:
		- Populates location_data_df with concatenated rows.
		"""

		weekly_dfs = []

		for week in weeks:
			filepath = self.DATA_PATH / f"tracking_week_{week}.parquet"
			df = self._load_parquet(filepath)
			if df is not None:
				weekly_dfs.append(df)
			else:
				print(f"Skipping week {week} due to load error.")

		self.location_data_df = pd.concat(weekly_dfs, ignore_index=True)

	def get_data(self, weeks: Iterable[int] = range(1, 10)) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		"""
		Loads requested datasets and returns them as a tuple.

		Inputs:
		- weeks: Weeks of tracking data to pull.

		Outputs:
		- games_df, plays_df, players_df, location_data_df in that order.
		"""

		self._load_base_data()
		self._load_tracking_data(weeks=weeks)

		return (self.games_df, self.plays_df, self.players_df, self.location_data_df)


def main() -> None:
	"""
	Demonstrates loading a subset of weeks for quick inspection.

	Inputs:
	- None (uses defaults).

	Outputs:
	- Loads data into DataFrames for manual exploration.
	"""

	loader = RawDataLoader()
	games_df, plays_df, players_df, location_data_df = loader.get_data(weeks=[1, 2])


if __name__ == "__main__":
	main()
