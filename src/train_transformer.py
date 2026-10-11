#!/usr/bin/env python
"""
Trains and tunes a transformer model for man/zone classification on tracking data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
import torch.nn as nn
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback  # your existing callback
from ray.train import Checkpoint, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, get_device, prepare_data_loader, prepare_model
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.tune_config import TuneConfig
from torch.amp import autocast
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader, TensorDataset

from common.args import parse_args
from common.decorators import set_time_decorators_enabled, time_fcn
from common.paths import PROCESSED_DIR, PROJECT_ROOT
from models.transformer import create_transformer_model


def plot_training_results(train_losses: list[float], val_losses: list[float]) -> None:
	"""
	Plots training and validation losses over epochs.

	Inputs:
	- train_losses: List of training losses per epoch.
	- val_losses: List of validation losses per epoch.

	Outputs:
	- Saves a plot of training and validation losses in artifacts/.
	"""

	epochs = range(1, len(train_losses) + 1)

	# Plot losses
	plt.figure(figsize=(12, 5), dpi=200)
	plt.plot(epochs, train_losses, label="Training Loss")
	plt.plot(epochs, val_losses, label="Validation Loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.title("Training and Validation Loss")
	plt.legend()
	plt.tight_layout()
	plt.savefig(PROJECT_ROOT / "artifacts" / "training_validation_loss_history.png")


def set_seed(seed: int = 42) -> torch.Generator:
	"""
	Sets Python, NumPy, and Torch seeds for reproducible runs.

	Inputs:
	- seed: Seed value to apply.

	Outputs:
	- generator: Torch generator for deterministic DataLoader shuffling.
	"""

	# Set seeds
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


def forward_backpropagate_optimize_step(amp_dtype: torch.dtype, model: nn.Module, features: torch.Tensor, targets: torch.Tensor, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
	# For forward pass, allow mixed precision for speed in some operations
	# NOTE: PyTorch keeps some sensitive ops in FP32 for stability
	with autocast(device_type="cuda", dtype=amp_dtype):
		outputs = model(features)
		loss = loss_fn(outputs, targets)

	# Backpropagate and optimize
	# NOTE: We don't want the gradients in less than FP32, PyTorch would handle this for us but being explicit
	loss.backward()
	optimizer.step()

	return loss


@time_fcn
def train_epoch(train_loader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device, amp_dtype: torch.dtype, prof: profile) -> float:
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

		# Profiled training step
		if prof:
			with record_function("train_step"):
				loss = forward_backpropagate_optimize_step(amp_dtype, model, features, targets, loss_fn, optimizer)
			prof.step()
		else:
			loss = forward_backpropagate_optimize_step(amp_dtype, model, features, targets, loss_fn, optimizer)

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
	train_features = torch.load(PROCESSED_DIR / "features_training.pt")
	train_targets = torch.load(PROCESSED_DIR / "targets_training.pt")

	val_features = torch.load(PROCESSED_DIR / "features_val.pt")
	val_targets = torch.load(PROCESSED_DIR / "targets_val.pt")

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
	# Create profiler
	######################################################################
	if args.profile:
		log_path = PROJECT_ROOT / "log" / "profiler"
		log_path.mkdir(parents=True, exist_ok=True)
		prof = profile(
			activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
			schedule=schedule(
				wait=1,  # ignore first step (cold start)
				warmup=1,  # record but don't save
				active=3,  # record + save 3 steps
				repeat=2,  # repeat that pattern
			),
			record_shapes=True,
			profile_memory=True,
			with_stack=True,
		)
		prof.start()
	else:
		prof = None

	######################################################################
	# Train model (and evaluate)
	######################################################################
	# Init
	train_losses = []
	val_losses = []
	val_accuracies = []
	best_val_loss = float("inf")

	# If running in CI mode, reduce epochs for speed, we just want to ensure it can actually train, not train a whole model in testing here (for pipelines later)
	if args.ci:
		config["epochs"] = 5

	# Training loop
	train_losses = []
	val_losses = []
	for epoch in range(int(config["epochs"])):
		# Train
		avg_train_loss = train_epoch(train_loader, model, optimizer, loss_fn, device, amp_dtype, prof)
		train_losses.append(avg_train_loss)

		# Validate
		avg_val_loss, val_accuracy = validate_epoch(val_loader, model, loss_fn, device, amp_dtype)
		val_losses.append(avg_val_loss)
		val_accuracies.append(val_accuracy)

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
			# Update saved model
			best_model_path = PROJECT_ROOT / "data" / "training" / "transformer.pt"
			if avg_val_loss < best_val_loss:
				best_val_loss = avg_val_loss
				torch.save(model.state_dict(), best_model_path)

	if not args.tune:
		# Plot the training results (we don't want to do this with HPO because Ray records these as metrics)
		ARTIFACT_PATH = PROJECT_ROOT / "artifacts"
		ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
		plot_training_results(train_losses, val_losses)

	# Stop profiler
	if prof:
		prof.stop()
		print("Top CPU ops:")
		print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
		print("Top GPU (CUDA kernel) ops:")
		print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
		prof.export_chrome_trace(str(log_path / "trace_training.json"))


@time_fcn
def run_hpo(args: argparse.Namespace) -> None:
	mlflow_tracking_uri = "sqlite:///mlflow.db"
	experiment = "transformer"

	params = {
		"model_dim": tune.choice([32, 64, 96, 128]),
		"num_heads": tune.choice([2, 4, 8]),
		"num_layers": tune.choice([2, 3, 4, 6]),
		"dropout": tune.choice([0.0, 0.1, 0.2, 0.3]),
		"multiplier": tune.choice([2, 3, 4]),
		"lr": tune.loguniform(1e-5, 5e-3),
		"weight_decay": tune.loguniform(1e-6, 1e-2),
		"batch_size": 19200,
		"epochs": 100,
		"compile": 1,
	}

	if args.ci:
		total_trials = 1
		params["epochs"] = 5
	else:
		total_trials = 1000

	scheduler = ASHAScheduler(
		max_t=int(params["epochs"]),
		grace_period=20 if not args.ci else 1,
		reduction_factor=2,
	)

	# Build a Trainer factory; Tune will pass each sampled config in here.
	def trainer_factory(cfg: dict[str, Any]) -> TorchTrainer:
		return build_trainer(
			cfg,
			num_workers=args.num_workers,  # <-- 1 works on any machine
			use_gpu=args.use_gpu,  # <-- True to use GPU
			run_name="transformer_hpo",
			storage_path=os.path.abspath("./ray_results"),
		)

	tuner = Tuner(
		trainable=trainer_factory,
		param_space=params,
		tune_config=TuneConfig(
			metric="val_accuracy",
			mode="max",
			scheduler=scheduler,
			num_samples=total_trials,
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

	results = tuner.fit()
	best = results.get_best_result(metric="val_accuracy", mode="max")
	logging.info(f"Best config: {best.config}")

	output_file = PROJECT_ROOT / "data" / "training" / "model_params.json"
	output_file.parent.mkdir(parents=True, exist_ok=True)
	with open(output_file, "w") as f:
		json.dump(best.config, f, indent=4)


def train_loop_per_worker(config: dict[str, Any]) -> None:
	"""
	Ray Train entrypoint. Runs once per worker (process).
	- If num_workers=1: behaves like normal single-process training.
	- If num_workers>1: wraps model in DDP and shards dataloaders.
	"""
	# --- Repro / perf flags ---
	g = set_seed(42)

	amp_dtype = torch.float16
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True

	device = get_device()  # Ray-assigned device (cuda:0 per worker if use_gpu=True)

	# --- Model / optim ---
	model = create_transformer_model(config).to(device)

	# DDP wrapping happens here when world_size > 1
	model = prepare_model(model)

	# torch.compile: OK in single GPU; with DDP it can be finicky.
	# Make it configurable so you can disable quickly if it causes trouble.
	if int(config.get("compile", 1)) == 1:
		try:
			# If DDP gives issues, try fullgraph=False first.
			model = torch.compile(model, mode="default", fullgraph=True)
		except Exception as e:
			logging.warning(f"torch.compile failed; continuing uncompiled. Error: {e}")

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=float(config["lr"]),
		weight_decay=float(config["weight_decay"]),
		fused=True,
	)
	loss_fn = nn.CrossEntropyLoss()

	# --- Data ---
	train_features = torch.load(PROCESSED_DIR / "features_training.pt")
	train_targets = torch.load(PROCESSED_DIR / "targets_training.pt")
	val_features = torch.load(PROCESSED_DIR / "features_val.pt")
	val_targets = torch.load(PROCESSED_DIR / "targets_val.pt")

	train_loader = DataLoader(
		TensorDataset(train_features, train_targets),
		batch_size=int(config["batch_size"]),
		shuffle=True,
		generator=g,
		num_workers=0,
		pin_memory=True,
	)
	val_loader = DataLoader(
		TensorDataset(val_features, val_targets),
		batch_size=int(config["batch_size"]),
		shuffle=False,
		num_workers=0,
		pin_memory=True,
	)

	# Shard data across workers automatically (no-op if num_workers=1)
	train_loader = prepare_data_loader(train_loader)
	val_loader = prepare_data_loader(val_loader)

	# --- Resume from checkpoint if available ---
	start_epoch = 0
	best_val_loss = float("inf")

	ckpt = train.get_checkpoint()
	if ckpt is not None:
		state = ckpt.to_dict()
		start_epoch = int(state.get("epoch", -1)) + 1
		best_val_loss = float(state.get("best_val_loss", float("inf")))

		# model may be DDP-wrapped; checkpoint stores unwrapped state_dict below
		unwrapped = model.module if hasattr(model, "module") else model
		unwrapped.load_state_dict(state["model"])
		optimizer.load_state_dict(state["optimizer"])

	epochs = int(config["epochs"])

	# --- Train ---
	for epoch in range(start_epoch, epochs):
		avg_train_loss = train_epoch(
			train_loader=train_loader,
			model=model,
			optimizer=optimizer,
			loss_fn=loss_fn,
			device=device,
			amp_dtype=amp_dtype,
			prof=None,  # strongly recommend disabling profiler in multi-worker runs initially
		)

		avg_val_loss, val_accuracy = validate_epoch(
			val_loader=val_loader,
			model=model,
			loss_fn=loss_fn,
			device=device,
			amp_dtype=amp_dtype,
		)

		best_val_loss = min(best_val_loss, float(avg_val_loss))

		metrics = {
			"epoch": epoch,
			"train_loss": float(avg_train_loss),
			"val_loss": float(avg_val_loss),
			"val_accuracy": float(val_accuracy),
		}

		# Save checkpoint every epoch (Tune/Train can pick best via metric)
		unwrapped = model.module if hasattr(model, "module") else model
		checkpoint = Checkpoint.from_dict(
			{
				"epoch": epoch,
				"best_val_loss": best_val_loss,
				"model": unwrapped.state_dict(),
				"optimizer": optimizer.state_dict(),
			}
		)

		train.report(metrics, checkpoint=checkpoint)


def build_trainer(
	config: dict[str, Any],
	*,
	num_workers: int,
	use_gpu: bool,
	run_name: str,
	storage_path: str,
) -> TorchTrainer:
	return TorchTrainer(
		train_loop_per_worker=train_loop_per_worker,
		train_loop_config=config,
		scaling_config=ScalingConfig(
			num_workers=int(num_workers),
			use_gpu=bool(use_gpu),
		),
		run_config=RunConfig(
			name=run_name,
			storage_path=storage_path,
			failure_config=FailureConfig(fail_fast=True),
		),
	)


def run_train(args: argparse.Namespace) -> None:
	# Load fixed params
	with open(PROJECT_ROOT / "data" / "training" / "model_params.json") as f:
		config = json.load(f)

	# Add a safety flag for compile toggling
	config.setdefault("compile", 1)

	if args.ci:
		config["epochs"] = min(int(config["epochs"]), 5)

	trainer = build_trainer(
		config,
		num_workers=args.num_workers,  # 1 works everywhere
		use_gpu=args.use_gpu,  # True if you want GPU
		run_name="transformer_train",
		storage_path=os.path.abspath("./ray_results"),
	)

	result = trainer.fit()
	best_ckpt = result.checkpoint
	if best_ckpt is None:
		raise RuntimeError("No checkpoint returned from Ray Train run.")

	state = best_ckpt.to_dict()
	model_state = state["model"]

	# Save to your existing location
	best_model_path = PROJECT_ROOT / "data" / "training" / "transformer.pt"
	best_model_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(model_state, best_model_path)

	logging.info(f"Saved model state_dict to: {best_model_path}")
	logging.info(f"Final metrics: {result.metrics}")

	# If you still want local plots, you can plot from result.metrics_dataframe if desired.


@time_fcn
def main() -> None:
	"""
	Entry point to run a single trial or hyperparameter search.

	Inputs:
	- CLI flags for tuning and profiling.

	Outputs:
	- Trains models and persists artifacts to disk.
	"""

	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

	args = parse_args()

	# Recommended defaults:
	if not hasattr(args, "num_workers"):
		args.num_workers = 1
	if not hasattr(args, "use_gpu"):
		args.use_gpu = torch.cuda.is_available()

	# Timing decorators
	set_time_decorators_enabled(bool(args.profile))
	logging.info(f"Timing decorators {'enabled' if args.profile else 'disabled'}")

	script_dir = Path(__file__).resolve().parent
	ray.init(
		_metrics_export_port=5001,
		runtime_env={"working_dir": str(script_dir)},
	)

	if args.tune:
		run_hpo(args)
	else:
		run_train(args)


if __name__ == "__main__":
	main()
