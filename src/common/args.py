#!/usr/bin/env python
"""
Parses shared CLI arguments used by training scripts.
"""

import argparse


def parse_args() -> argparse.Namespace:
	"""
	Defines and parses common command-line flags.

	Inputs:
	- None (relies on sys.argv).

	Outputs:
	- args: Namespace containing tune/profile/ci flags.
	"""

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
