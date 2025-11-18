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
import random

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

from models.transformer import ManZoneTransformer
from common.decorators import time_fcn
from common.paths import PROJECT_ROOT, SAVE_DIR

def set_seed(seed: int = 42):
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
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

@time_fcn
def train_epoch(train_loader: DataLoader, val_loader: DataLoader, model, optimizer, loss_fn, device) -> tuple[float, float]:
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
        return avg_train_loss, avg_val_loss, val_accuracy


@time_fcn
def main():

    # Init logging and save path
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define model and device
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ManZoneTransformer(
        feature_len=5,          # num of input features (x, y, v_x, v_y, defense)
        model_dim=64,           # TBD with tune
        num_heads=2,            # TBD with tune
        num_layers=4,           # TBD with tune
        dim_feedforward=64 * 4, # model_dim * number of layers
        dropout=0.1,            # TBD with tune
        output_dim=2            # man or zone classification
    ).to(device)

    # Set optimizer and loss fcn
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    early_stopping_patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    num_epochs = 50

    # Init random
    g = torch.Generator()    # Creates a generator that fixes the shuffle in torch Dataloader
    g.manual_seed(42)

    # Create data loaders
    train_features = torch.load(SAVE_DIR / f"features_training.pt")
    train_targets = torch.load(SAVE_DIR / f"targets_training.pt")
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=64,
        shuffle=True,                       # We want random mini batches so GD doesn't overfit to specific ordering patterns, lets shuffle
        generator=g,                        # Fixes the shuffle
        num_workers=0,                      # Eliminate worker non-determinism
        pin_memory=True,                    # Batches are allocated on page-locked ("pinned") memory on the host, allows GPU driver to perform faster async DMA
    )  
    
    val_features = torch.load(SAVE_DIR /  f"features_val.pt")
    val_targets = torch.load(SAVE_DIR /  f"targets_val.pt")
    val_loader = DataLoader(
        TensorDataset(val_features, val_targets),
        batch_size=64,
        shuffle=False,                      # In eval we aren't updating the weights, so it doesn't really matter if we imply ordering or not
        num_workers=0,                      # Eliminate worker non-determinism
        pin_memory=True,                    # Batches are allocated on page-locked ("pinned") memory on the host, allows GPU driver to perform faster async DMA
    )       

    # Train with early stoppping
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):

        avg_train_loss, avg_val_loss, val_accuracy = train_epoch(train_loader, val_loader, model, optimizer, loss_fn, device)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logging.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Early stopping
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

if __name__ == "__main__":
  main()