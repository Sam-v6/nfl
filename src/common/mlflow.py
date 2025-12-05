#!/usr/bin/env python
"""
Lightweight helper to configure MLflow tracking for experiments.
"""

import mlflow


def setup_mlflow(experiment_name: str = "random-experiement", tracking_uri: str = "file:./mlruns") -> None:
	"""
	Initializes MLflow tracking URI and experiment.

	Inputs:
	- experiment_name: Name of the experiment to log under.
	- tracking_uri: Backend store location.

	Outputs:
	- Sets global MLflow state for subsequent logging calls.
	"""

	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(experiment_name)
