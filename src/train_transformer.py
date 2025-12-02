#!/usr/bin/env python
"""
Trains and tunes a transformer model for man/zone classification on tracking data.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import ray
import torch
import torch.nn as nn
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import FailureConfig, RunConfig, TuneConfig, Tuner
from ray.tune.schedulers import ASHAScheduler
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset

from common.args import parse_args
from common.decorators import set_time_decorators_enabled, time_fcn
from common.paths import PROJECT_ROOT, SAVE_DIR
from models.transformer import create_transformer_model


def set_seed(seed: int = 42) -> torch.Generator:
	"""
	Sets Python, NumPy, and Torch seeds for reproducible runs.

	Inputs:
	- seed: Seed value to apply.

	Outputs:
	- generator: Torch generator for deterministic DataLoader shuffling.
	"""
	# Python & NumPy
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	g = torch.Generator()  # Creates a generator that fixes the shuffle in torch Dataloader
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
def train_epoch(train_loader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device, amp_dtype: torch.dtype) -> float:
	"""
	Runs one training epoch over the provided dataloader.

	Inputs:
	- train_loader: Batches of training tensors.
	- model: Transformer model to optimize.
	- optimizer: Optimizer instance.
	- loss_fn: Loss function.
	- device: Target device for computation.
	- amp_dtype: Mixed precision dtype for autocast.

	Outputs:
	- avg_train_loss: Mean loss across the epoch.
	"""
	# Training
	model.train()
	running_loss = 0.0
	for features, targets in train_loader:
		# Transfer from CPU to GPU, non_blocking for pinned memory
		features = features.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		# Zero gradients
		optimizer.zero_grad(set_to_none=True)

		# For forward pass, allow mixed precision for speed in some operations
		# NOTE: PyTorch keeps some sensitive ops in FP32 for stability
		with autocast(device_type="cuda", dtype=amp_dtype):
			outputs = model(features)
			loss = loss_fn(outputs, targets)

		# Backpropagate and optimize
		# NOTE: We don't want the gradients in less than FP32, PyTorch would handle this for us but being explicit
		loss.backward()
		optimizer.step()

		# Record running loss
		running_loss += loss.detach().item() * features.size(0)

	# Record running metrics
	avg_train_loss = running_loss / len(train_loader.dataset)

	# Return losses
	return avg_train_loss


@time_fcn
def validate_epoch(val_loader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: torch.device, amp_dtype: torch.dtype) -> tuple[float, float]:
	"""
	Validates the model on a held-out split.

	Inputs:
	- val_loader: Batches of validation tensors.
	- model: Transformer being evaluated.
	- loss_fn: Loss function.
	- device: Target device for computation.
	- amp_dtype: Mixed precision dtype for autocast.

	Outputs:
	- avg_val_loss: Mean validation loss.
	- val_accuracy: Classification accuracy.
	"""
	# Validation
	model.eval()
	val_running_loss = 0.0
	correct = 0
	with torch.no_grad():
		for val_features_batch, val_targets_batch in val_loader:
			# Transfer from CPU to GPU, non_blocking for pinned memory
			val_features_batch = val_features_batch.to(device, non_blocking=True)
			val_targets_batch = val_targets_batch.to(device, non_blocking=True)

			# AMP
			with autocast(device_type="cuda", dtype=amp_dtype):
				val_outputs = model(val_features_batch)
				val_loss = loss_fn(val_outputs, val_targets_batch)
				_, predicted = torch.max(val_outputs, 1)

			# Grab metrics
			val_running_loss += val_loss.item() * val_features_batch.size(0)
			correct += (predicted == val_targets_batch).sum().item()

	# Record running metrics
	avg_val_loss = val_running_loss / len(val_loader.dataset)
	val_accuracy = correct / len(val_loader.dataset)

	# Return losses
	return avg_val_loss, val_accuracy


@time_fcn
def run_trial(config: dict[str, float | int], args: argparse.Namespace) -> None:
	"""
	Executes a single training run with the supplied hyperparameters.

	Inputs:
	- config: Hyperparameters for the transformer and training loop.
	- args: Command-line arguments controlling CI/tuning behavior.

	Outputs:
	- Trains the model and optionally reports metrics to Ray Tune.
	"""

	# Set seeds
	g = set_seed(42)

	######################################################################
	# Set things that allow for speed-up in training (valid on Ampere+ GPUs)
	######################################################################
	# NOTE: Importing torch here because Ray serializes the fcn, if we allow the global scope torch it fails in trying to serialize the global chain
	import torch

	# Use mixed precision with FP16 for speed for forward pass (see train_epoch)
	amp_dtype = torch.float16

	# Set TensorFloat-32 (TF32) mode for matmul and cudnn (speeds up training on Ampere+ GPUs with minimal impact on accuracy)
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True

	# Set device
	device = torch.device("cuda")  # this is the Ray-assigned GPU (Ray sets CUDA_VISIBLE_DEVICES)

	######################################################################
	# Create model, loss, optimizer
	######################################################################
	# Create transformer with given config, move to GPU, and compile model for better inference (will save as compiled model)
	model = create_transformer_model(config)
	model = model.to(device)
	model = torch.compile(
		model,
		mode="default",  # default, reduces overhead, generally stable
		fullgraph=True,  # enable full graph fusion, helps reduce kernel launch overhead
	)

	# Set optimizer and loss fcn
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=float(config["lr"]),
		weight_decay=float(config["weight_decay"]),
		fused=True,  # All operations on single CUDA kernel for speed
	)
	loss_fn = nn.CrossEntropyLoss()

	######################################################################
	# Create dataloaders
	######################################################################
	# Load in data and create tensor datasets
	train_features = torch.load(SAVE_DIR / "features_training.pt")
	train_targets = torch.load(SAVE_DIR / "targets_training.pt")

	val_features = torch.load(SAVE_DIR / "features_val.pt")
	val_targets = torch.load(SAVE_DIR / "targets_val.pt")

	# Create data loaders for batching
	train_loader = DataLoader(
		TensorDataset(train_features, train_targets),
		batch_size=int(config["batch_size"]),
		shuffle=True,  # We want random mini batches so GD doesn't overfit to specific ordering patterns, lets shuffle
		generator=g,  # Fixes the shuffle
		num_workers=0,  # Eliminate worker non-determinism
		pin_memory=True,  # Batches are allocated on page-locked ("pinned") memory on the host, allows GPU driver to perform faster async DMA
	)

	val_loader = DataLoader(
		TensorDataset(val_features, val_targets),
		batch_size=int(config["batch_size"]),
		shuffle=False,  # In eval we aren't updating the weights, so it doesn't really matter if we imply ordering or not
		num_workers=0,  # Eliminate worker non-determinism
		pin_memory=True,  # Batches are allocated on page-locked ("pinned") memory on the host, allows GPU driver to perform faster async DMA
	)

	######################################################################
	# Train model (and evaluate)
	######################################################################
	# Init
	train_losses = []
	val_losses = []
	val_accuracies = []
	early_stopping_patience = 20
	best_val_loss = float("inf")
	epochs_no_improve = 0

	# If running in CI mode, reduce epochs for speed, we just want to ensure it can actually train, not train a whole model in testing here (for pipelines later)
	if args.ci:
		config["epochs"] = 5

	for epoch in range(int(config["epochs"])):
		# Train
		avg_train_loss = train_epoch(train_loader, model, optimizer, loss_fn, device, amp_dtype)
		train_losses.append(avg_train_loss)

		# Validate
		avg_val_loss, val_accuracy = validate_epoch(val_loader, model, loss_fn, device, amp_dtype)

		# Info
		logging.info(f"Epoch [{epoch + 1}/{int(config['epochs'])}]")
		logging.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

		if args.tune:
			######################################################################
			# Tune logging (with MLflow callback this is all mirrored there as well)
			# NOTE: By calling tune.report here effectively once per epoch, that becomes our time scale!
			######################################################################
			# Record metrics
			metrics = {
				"val_accuracy": val_accuracy,
				"val_loss": float(avg_val_loss),
				"train_loss": float(avg_train_loss),
			}
			tune.report(metrics)
		else:
			# Record metrics
			val_losses.append(avg_val_loss)
			val_accuracies.append(val_accuracy)

			# Adding early stopping check (effort to prevent overfitting)
			best_model_path = PROJECT_ROOT / "data" / "training" / "transformer.pt"
			if avg_val_loss < best_val_loss:
				best_val_loss = avg_val_loss
				epochs_no_improve = 0
				torch.save(model.state_dict(), best_model_path)
			else:
				epochs_no_improve += 1
				if epochs_no_improve >= early_stopping_patience:
					logging.info("Early stopping triggered")
					break


@time_fcn
def run_hpo(args: argparse.Namespace) -> None:
	"""
	Launches Ray Tune to search over transformer hyperparameters.

	Inputs:
	- args: Command-line options controlling CI/tuning behavior.

	Outputs:
	- Writes best config to disk and logs metrics to MLflow.
	"""

	######################################################################
	# Start parent HPO, MLflow session
	######################################################################
	mlflow_tracking_uri = "sqlite:///mlflow.db"
	experiment = "transformer"

	######################################################################
	# Define search space and scheduler
	######################################################################
	transformer_params = {
		# Varying model shape
		"model_dim": tune.choice([32, 64, 96, 128]),
		"num_heads": tune.choice([2, 4, 8]),
		"num_layers": tune.choice([2, 3, 4, 6]),
		"dropout": tune.choice([0.0, 0.1, 0.2, 0.3]),
		"multiplier": tune.choice([2, 3, 4]),
		# Training
		"lr": tune.loguniform(1e-5, 5e-3),
		"weight_decay": tune.loguniform(1e-6, 1e-2),
		"batch_size": 19200,  # Configured for L4 GPU utilization
		# Epochs
		"epochs": 100,
	}
	params = transformer_params

	# Async Successive Halfing Scheduler (ASHA)
	# Instead of running all trials for all epochs, it allocates more resources to promising ones and kills of bad ones early
	scheduler = ASHAScheduler(
		max_t=params["epochs"],  # Max amount of "things" on our whatever our scale is (since we call tune.report once per epoch this max epochs per trial)
		grace_period=20,  # Allow for 20 epochs before pruning (so each trial gets at least this many epochs)
		reduction_factor=2,  # ASHA keeps about 50% of the top trials each time it prunes
	)

	######################################################################
	# Build tuner; pass MLflow context and PARENT RUN ID to workers via env vars
	######################################################################
	trainable = tune.with_parameters(run_trial, args=args)  # Allows each training run to have any specific params
	tuner = Tuner(
		tune.with_resources(trainable, resources={"cpu": 16, "gpu": 1}),  # Gives 16 CPU and one GPU per trial
		param_space=params,
		tune_config=TuneConfig(
			metric="val_accuracy",
			mode="max",
			scheduler=scheduler,
			num_samples=1,  # total trials, set to 1 for CI, modify here for HPO runs
		),
		run_config=RunConfig(
			name="transformer_hpo",
			storage_path=os.path.abspath("./ray_results"),
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

	# Save file to best config
	output_file = PROJECT_ROOT / "data" / "training" / "model_params.json"
	with open(output_file, "w") as json_file:
		json.dump(best.config, json_file, indent=4)  # indent for pretty-printing


@time_fcn
def main() -> None:
	"""
	Entry point to run a single trial or hyperparameter search.

	Inputs:
	- CLI flags for tuning and profiling.

	Outputs:
	- Trains models and persists artifacts to disk.
	"""

	# Set logging
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
		# Set the runtime env to only ship src/, the whole repo that has a bunch of data
		script_dir = Path(__file__).resolve().parent  # Directory containing train_transformer.py (i.e., src/)
		ray.init(
			# Set the system config for the metrics exporter to pick a free port
			_metrics_export_port=5001,
			# Set the runtime to only bundle the src dir
			runtime_env={
				"working_dir": str(script_dir),
			},
		)
		run_hpo(args)

	# We are doing a single run with fixed model parameters located at nfl/data/training/model_params.json
	else:
		with open(PROJECT_ROOT / "data" / "training" / "model_params.json") as file:
			model_params_map = json.load(file)
		run_trial(model_params_map, args)


if __name__ == "__main__":
	main()
