#!/usr/bin/env python

"""
Parses input arguments to then act on in code
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Use Ray Tune to search hyperparameters instead of a single training run",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable @time_fcn timing decorators for profiling",
    )

    parser.add_argument(
        "--ci",
        action="store_true",
        help="Enable CI mode with reduced epochs for faster training during pipeline",
    )
    return parser.parse_args()