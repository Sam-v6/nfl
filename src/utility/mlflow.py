#!/usr/bin/env python

"""
Module: data_loader.py
Description: Functions for loading and preprocessing test data for the thruster analysis pipeline.

Author: Syam Evani
Created: 2025-11-02
"""

# MLflow
import mlflow

def setup_mlflow(experiment_name: str = "random-experiement", tracking_uri: str = "file:./mlruns"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
