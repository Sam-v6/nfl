#!/usr/bin/env python

"""
Trains transformer model on location tracking data

Requires that create_features.py has already been ran and produced:
- features_training.pt
- features_val.pt
- targets_training.pt
- targets_val.pt

Will train 50 epochs (unless it early stops) and produce model.pth
"""

import os
import logging
import math
from pathlib import Path
import random
import random
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib

# Ray Tune
import ray
from ray import train, tune, air
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray.tune import Tuner, RunConfig, TuneConfig, FailureConfig, Checkpoint
from ray.tune import Tuner, RunConfig, TuneConfig, FailureConfig
from ray.tune.schedulers import ASHAScheduler

from models.transformer import ManZoneTransformer
from common.decorators import set_time_decorators_enabled, time_fcn
from common.paths import PROJECT_ROOT, SAVE_DIR
from common.args import parse_args

def set_seed(seed: int = 42) -> torch.Generator:
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()    # Creates a generator that fixes the shuffle in torch Dataloader
    g.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch CPU & CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Keep cudnn deterministic, but allow normal algorithms
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # DO NOT enforce deterministic algorithms globally
    # torch.use_deterministic_algorithms(False)

    return g


@time_fcn
def train_epoch(train_loader: DataLoader, model, optimizer, loss_fn, device) -> float:
        # Training
        model.train()
        running_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        avg_train_loss = running_loss / len(train_loader.dataset)

        # Return losses
        return avg_train_loss


@time_fcn
def validate_epoch(val_loader: DataLoader, model, loss_fn, device) -> tuple[float, float]:
        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for val_features_batch, val_targets_batch in val_loader:
                val_features_batch, val_targets_batch = val_features_batch.to(device), val_targets_batch.to(device)
                val_outputs = model(val_features_batch)
                val_loss = loss_fn(val_outputs, val_targets_batch)
                val_running_loss += val_loss.item() * val_features_batch.size(0)
                _, predicted = torch.max(val_outputs, 1)
                correct += (predicted == val_targets_batch).sum().item()
        avg_val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct / len(val_loader.dataset)

        # Return losses
        return avg_val_loss, val_accuracy


@time_fcn
def train_trial(config, args):

    # Set seeds
    g = set_seed(42)

    ######################################################################
    # Create model, loss, optimizer
    ######################################################################
    device = torch.device("cuda")   # this is the Ray-assigned GPU (Ray sets CUDA_VISIBLE_DEVICES)

    # Defining ManZoneTransformer params, initializing optimizer and loss_fn
    model = ManZoneTransformer(
        feature_len=5,                                                          # num of input features (x, y, v_x, v_y, defense)
        model_dim=int(config["model_dim"]),                                     # from ray tune or loaded
        num_heads=int(config["num_heads"]),                                     # from ray tune or loaded
        num_layers=int(config["num_layers"]),                                   # from ray tune or loaded
        dim_feedforward=int(config["model_dim"]) * int(config["num_layers"]),   # from ray tune or loaded
        dropout=float(config["dropout"]),                                       # 10% dropout to prevent overfitting... iterate as model becomes more complex (industry std is higher, i believe)
        output_dim=2                                                            # man or zone classification
    ).to(device)

    # Set optimizer and loss fcn
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    ######################################################################
    # Create dataloaders
    ######################################################################
    # Load in data and create tensor datasets
    train_features = torch.load(SAVE_DIR / f"features_training.pt")
    train_targets = torch.load(SAVE_DIR / f"targets_training.pt")

    val_features = torch.load(SAVE_DIR / f"features_val.pt")
    val_targets = torch.load(SAVE_DIR / f"targets_val.pt")

    # Create data loaders for batching
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=int(config["batch_size"]),
        shuffle=True,                       # We want random mini batches so GD doesn't overfit to specific ordering patterns, lets shuffle
        generator=g,                        # Fixes the shuffle
        num_workers=0,                      # Eliminate worker non-determinism
        pin_memory=True,                    # Batches are allocated on page-locked ("pinned") memory on the host, allows GPU driver to perform faster async DMA
        )  
    
    val_loader = DataLoader(
        TensorDataset(val_features, val_targets),
        batch_size=int(config["batch_size"]),
        shuffle=False,                      # In eval we aren't updating the weights, so it doesn't really matter if we imply ordering or not
        num_workers=0,                      # Eliminate worker non-determinism
        pin_memory=True,                    # Batches are allocated on page-locked ("pinned") memory on the host, allows GPU driver to perform faster async DMA
        )       

    ######################################################################
    # Train model (and evaluate)
    ######################################################################
    # Init
    train_losses = []
    val_losses = []
    val_accuracies = []
    early_stopping_patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(int(config["epochs"])):

        # Train
        avg_train_loss = train_epoch(train_loader, model, optimizer, loss_fn, device)
        train_losses.append(avg_train_loss)

        # Validate
        avg_val_loss, val_accuracy = validate_epoch(val_loader, model, loss_fn, device)

        # Info
        logging.info(f"Epoch [{epoch + 1}/{int(config['epochs'])}]")
        logging.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if args.tune:
            ######################################################################
            # Tune logging (with MLflow callback this is all mirrored there as well)
            # NOTE: By calling tune.report here effectively once per epoch, that becomes our time scale!
            ######################################################################
            session   = tune.get_context()
            trial_dir = Path(session.get_trial_dir())
            best_model_path = trial_dir / "best_model.pth"

            # Record metrics
            metrics = {
                "val_accuracy": val_accuracy,
                # "val_loss": float(avg_val_loss),
                # "train_loss": float(avg_train_loss),
            }

            # Report metrics and save checkpoint if applicable (checkpoint every n epochs and don't have redudant checkpoints if using workers via train)
            checkpoint = None
            should_checkpoint = epoch % config.get("checkpoint_freq", 1) == 0
            if should_checkpoint:
                ckpt_dir  = trial_dir / f"ckpt_e{epoch:04d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / "model.pth")
                checkpoint = Checkpoint.from_directory(str(ckpt_dir))
                tune.report(metrics, checkpoint=checkpoint)
            else:
                tune.report(metrics)
        
        else:
            # Record metrics
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            # Adding early stopping check (effort to prevent overfitting)
            best_model_path = PROJECT_ROOT / "data" / "training" / "best_model.pth"
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    logging.info(f"Early stopping triggered")
                    break


@time_fcn       
def run_HPO(args):

    ######################################################################
    # Start parent HPO, MLflow session
    ######################################################################
    mlflow_tracking_uri = "sqlite:///mlflow.db"
    experiment   = "transformer"

    ######################################################################
    # Define search space and scheduler
    ######################################################################
    transformer_params = {
        # Varying model shape
        "model_dim": tune.choice([32, 64, 128, 256]),
        "num_heads": tune.choice([2, 4]),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "dropout": tune.choice([0.0, 0.1, 0.2]) ,

        # Training
        "batch_size": tune.choice([32, 64, 128]),

        # Epochs / checkpointing
        "epochs": 50,
        "checkpoint_freq": 10,
    }

    params=transformer_params

    # Async Successive Halfing Scheduler (ASHA)
    # Instead of running all trials for all epochs, it allocates more resources to promising ones and kills of bad ones early
    scheduler = ASHAScheduler(
        max_t=params["epochs"],                     # Max amount of "things" on our whatever our scale is (since we call tune.report once per epoch this max epochs per trial)
        grace_period=params["checkpoint_freq"]+1,   # Allow for 6 epochs each trial until we kill it
        reduction_factor=2,                         # ASHA keeps about 50% of the top trials each time it prunes
    )
    
    ######################################################################
    # Build tuner; pass MLflow context and PARENT RUN ID to workers via env vars
    ######################################################################
    trainable = tune.with_parameters(train_trial, args=args)  # Allows each training run to have any specific params
    tuner = Tuner(
        tune.with_resources(trainable, resources={"cpu": 4, "gpu": 1}), # Gives 4 CPU and one GPU per trial
        param_space=params,
        tune_config=TuneConfig( 
            metric="val_accuracy",
            mode="max",
            scheduler=scheduler,
            num_samples=10,             # total trials
        ),
        run_config=RunConfig(
            name="transformer_hpo",
            storage_path=os.path.abspath("./log/ray_results"),
            failure_config=FailureConfig(fail_fast=True),
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name=experiment,
                    save_artifact=True,
                )
            ],
        ),
    )

    ######################################################################
    # Execute HPO
    ######################################################################
    results = tuner.fit()
    best = results.get_best_result(metric="val_accuracy", mode="max")
    print("Best config:", best.config)


@time_fcn
def main():

    # Set logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Get input args
    args = parse_args()

    # Enable/disable timing decorators
    if args.profile:
        set_time_decorators_enabled(True)
        logging.info("Timing decorators enabled")
    else:
        set_time_decorators_enabled(False)
        logging.info("Timing decorators disabled")

    # We are doing HPO with Ray
    if args.tune:
        # Disable some warnings about GPUs if we don't use them (we are always using GPUs for training here)
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

        # Set the runtime env to only ship src/, the whole repo that has a bunch of data
        script_dir = Path(__file__).resolve().parent         # Directory containing train_transformer.py (i.e., src/)
        ray.init(
            runtime_env={
                "working_dir": str(script_dir),
            }
        )
        run_HPO(args)

    # We are doing a single run with fixed model parameters located at nfl/data/training/model_params.json
    else:
        with open(PROJECT_ROOT / "data" / "training" / "model_params.json", 'r') as file:
            model_params_map = json.load(file)
        train_trial(model_params_map, args)


if __name__ == "__main__":
    main()