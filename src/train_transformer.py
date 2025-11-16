#!/usr/bin/env python

"""
Trains transformer model on location tracking data

Requires that create_features.py has already been ran and produced:
- features_training.pt
- features_val.pt
- targets_training.pt
- targets_val.pt

Will train 30 epochs (unless it early stops) and produce model.pth
"""

import os
import logging
import math

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

from models.transformer import ManZoneTransformer
from common.decorators import time


@time
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


@time
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    save_path = os.path.join(os.getenv('NFL_HOME'), 'data', 'processed')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # defining ManZoneTransformer params, initializing optimizer and loss_fn
    model = ManZoneTransformer(
        feature_len=5,    # num of input features (x, y, v_x, v_y, defense)
        model_dim=64,     # experimented with 96 & 128... seems best
        num_heads=2,      # 2 seems best (but may have overfit when tried 4... may be worth iterating & increasing dropout)
        num_layers=4,
        dim_feedforward=64 * 4,
        dropout=0.1,      # 10% dropout to prevent overfitting... iterate as model becomes more complex (industry std is higher, i believe)
        output_dim=2      # man or zone classification
    ).to(device)

    batch_size = 64
    learning_rate = 1e-3

    batch_size = 64
    learning_rate = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    early_stopping_patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    num_epochs = 30

    # loading in data & placing into DataLoader object
    train_features = torch.load(os.path.join(save_path, f"features_training.pt"))
    train_targets = torch.load(os.path.join(save_path, f"targets_training.pt"))
    val_features = torch.load(os.path.join(save_path, f"features_val.pt"))
    val_targets = torch.load(os.path.join(save_path, f"targets_val.pt"))

    # move data to device (think it needs to be consistent)
    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    val_features = val_features.to(device)
    val_targets = val_targets.to(device)

    # Create data laoders
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # manually placing an early stopping method... will iterate on the exact value (currently 5) but want to prevent overfitting
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

        # adding early stopping check (effort to prevent overfitting)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # saving the best model
            torch.save(model.state_dict(), os.path.join(save_path, f"model.pth"))

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                logging.info(f"Early stopping triggered")
                break

if __name__ == "__main__":
  main()