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
import joblib # Save pkl

# Ray Tune
import ray
from ray import train, tune, air
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray.train import Checkpoint
from ray.tune import Tuner, RunConfig, TuneConfig, FailureConfig
from ray.tune.schedulers import ASHAScheduler

from models.transformer import ManZoneTransformer
from common.decorators import time_fcn
from common.paths import PROJECT_ROOT, SAVE_DIR

def set_seed(seed: int = 42) -> torch.Generator:
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(base_seed)
    g = torch.Generator()    # Creates a generator that fixes the shuffle in torch Dataloader
    g.manual_seed(base_seed)

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
def validate_epoch(val_loader: DataLoader, model, optimizer, loss_fn, device) -> tuple[float, float]:
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
def train_trial(config):

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
        avg_val_loss, val_accuracy = validate_epoch(val_loader, model, optimizer, loss_fn, device)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Info
        logging.info(f"Epoch [{epoch+1}/{int(config["epochs"])}]")
        logging.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Adding early stopping check (effort to prevent overfitting)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # saving the best model
            torch.save(model.state_dict(), SAVE_DIR / f"model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                logging.info(f"Early stopping triggered")
                break

        ######################################################################
        # Tune logging (with MLflow callback this is all mirrored there as well)
        # NOTE: By calling tune.report here effectively once per epoch, that becomes our time scale!
        ######################################################################
        # Report metrics and save checkpoint if applicable (checkpoint every n epochs and don't have redudant checkpoints if using workers via train)
        checkpoint = None
        should_checkpoint = epoch % config.get("checkpoint_freq", 1) == 0

        # NOTE: In standard DDP training, where the model is the same across all ranks, only the global rank 0 worker needs to save and report the checkpoint
        if should_checkpoint: # add in tune.get_context().get_world_rank() == 0 when workers implemented

            # Create the checkpoint dir
            session   = tune.get_context()
            trial_dir = Path(session.get_trial_dir())
            ckpt_dir = trial_dir / f"ckpt_e{epoch:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Save the model
            session = tune.get_context()
            trial_dir = Path(session.get_trial_dir())
            best_model_path = trial_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)

            # Save scaler
            #joblib.dump(scaler, ckpt_dir / "standard_scaler.pkl")

            # Create checkpoint
            ckpt = Checkpoint.from_directory(str(ckpt_dir))

        # We want to report metrics every epoch regardless if we are checkpointing
        metrics = {
            "val_accuracy": float(val_accuracies[-1]),
        }
        tune.report(metrics, checkpoint=ckpt)


@time_fcn
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define model and device
    set_seed(42)

    # Directory containing train_transformer.py (i.e., src/)
    script_dir = Path(__file__).resolve().parent

    ray.init(
        runtime_env={
            # Only ship src/, not the whole repo
            "working_dir": str(script_dir),
        }
    )

    # Run hyperparameter optimization
    run_HPO()

@time_fcn       
def run_HPO():

    ######################################################################
    # Start parent HPO, MLflow session
    ######################################################################
    mlflow_tracking_uri = f"file:{os.path.abspath('./log/mlruns')}"  # absolute path
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
        "checkpoint_freq": 5,
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
    trainable = tune.with_parameters(train_trial)  # Allows each training run to have any specific params
    tuner = Tuner(
        tune.with_resources(trainable, resources={"cpu": 4, "gpu": 1}),                  # Gives 4 CPU and one GPU per trial
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

if __name__ == "__main__":
    main()