#!/usr/bin/env python

"""
Module: data_loader.py
Description: Functions for loading and preprocessing test data for the thruster analysis pipeline.

Author: Syam Evani
Created: 2025-11-02
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn

from load_data import RawDataLoader

# What do I care about
# Location 

def get_stats(


def main():

  # Get raw data
  rawLoader = RawDataLoader()
  games_df, plays_df, players_df, location_data_df = rawLoader.get_data(weeks=[i for i in range(1, 10)])

  pass

if __name__ == "__main__":
  main()