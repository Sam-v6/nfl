"""One train/evaluate loop for local runs and Ray trials, with portable resume."""

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from dataset import file_hash
from dataset_io import load_examples
from evaluate import _macro_from_cm, score_predictions
from model_bundle import save_bundle
from models.factory import create_coverage_model, forward_model

DEFAULTS = {
	"model_dim": 64,
	"num_heads": 4,
	"num_layers": 2,
	"multiplier": 4,
	"dropout": 0.1,
	"num_classes": 7,
	"feature_len": 5,
	"epochs": 30,
	"batch_size": 512,
	"lr": 0.0003,
	"weight_decay": 0.01,
	"seed": 42,
	"device": "auto",
	"precision": "auto",
	"compile": False,
	"patience": 10,
	"input_pipeline": "loader",
	"model_type": "spatial",
	"fit_season": None,
	"class_weighting": False,
	"inference_precision": "fp32",
	"tf32": False,
	"cpu_threads": 4,
}


def play_sampling_weights(index: pd.DataFrame) -> torch.Tensor:
	"""Give each season/game/play equal mass across endpoints and mirrors."""
	counts = index.groupby(["season", "gameId", "playId"])["playId"].transform("size")
	return torch.from_numpy(1.0 / counts.to_numpy(dtype="float64"))


def select_device(name: str) -> torch.device:
	if name not in ("cpu", "cuda", "auto"):
		raise ValueError("device must be cpu, cuda or auto")
	if name == "cuda" and not torch.cuda.is_available():
		raise ValueError("CUDA requested but unavailable; run with host GPU access")
	return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def _score(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict[str, Any]:
	probs = logits.float().softmax(dim=-1).cpu().numpy()
	y = targets.cpu().numpy()
	if num_classes == 7:
		return score_predictions(y, probs)
	predicted = probs.argmax(axis=1)
	cm = np.bincount(y * 2 + predicted, minlength=4).reshape(2, 2)
	return {"accuracy": float((predicted == y).mean()), "macro_f1": _macro_from_cm(cm), "log_loss": float(-np.log(probs[np.arange(len(y)), y].clip(1e-15, 1)).mean()), "confusion": cm.tolist()}


def train_model(config: dict[str, Any], manifest_path: Path, run_dir: Path, reporter: Callable[[dict[str, Any], Path], None] | None = None) -> Path:
	settings = {**DEFAULTS, **config}
	if not 1 <= int(settings["cpu_threads"]) <= 32:
		raise ValueError("CPU thread count must be 1–32")
	torch.set_num_threads(int(settings["cpu_threads"]))
	device = select_device(str(settings["device"]))
	epochs = int(settings["epochs"])
	batch_size = int(settings["batch_size"])
	if not 1 <= epochs <= 100 or batch_size < 1 or settings["num_classes"] not in (2, 7):
		raise ValueError("Require 1–100 epochs, positive batch size, and 2 or 7 classes")
	precision = str(settings["precision"])
	if precision == "auto":
		precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
	if precision not in ("fp32", "bf16", "fp16") or (device.type == "cpu" and precision != "fp32"):
		raise ValueError("Unsupported precision for device")
	if precision == "bf16" and not torch.cuda.is_bf16_supported():
		raise ValueError("GPU does not support BF16")
	amp_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
	resume = bool(settings.pop("resume", False))
	if settings["inference_precision"] != "fp32":
		raise ValueError("Model selection uses the FP32 inference reference")
	identity = {key: value for key, value in settings.items() if key not in ("epochs", "fitted_class_weights")}
	if settings.get("sampling", "endpoint") not in ("endpoint", "play"):
		raise ValueError("Sampling must be endpoint or play")
	x_train, mask_train, y_train, train_index = load_examples(manifest_path, "fit", season=settings["fit_season"])
	x_val, mask_val, y_val, _ = load_examples(manifest_path, "tune", offset=-1)
	if settings["model_type"] == "spatial":
		x_train, x_val = x_train[:, -1:].contiguous(), x_val[:, -1:].contiguous()
		mask_train, mask_val = mask_train[:, -1:].contiguous(), mask_val[:, -1:].contiguous()
	if settings["num_classes"] == 2:
		y_train = ((y_train == 0) | (y_train == 1) | (y_train == 5)).long()
		y_val = ((y_val == 0) | (y_val == 1) | (y_val == 5)).long()
	sampling_weights = play_sampling_weights(train_index) if settings.get("sampling") == "play" else None
	manifest = json.loads(manifest_path.read_text())
	run_dir.mkdir(parents=True, exist_ok=resume)
	started = time.monotonic()
	torch.manual_seed(int(settings["seed"]))
	if device.type == "cuda":
		torch.cuda.manual_seed_all(int(settings["seed"]))
		torch.backends.cuda.matmul.allow_tf32 = bool(settings["tf32"])
		torch.backends.cudnn.allow_tf32 = bool(settings["tf32"])
	generator = torch.Generator().manual_seed(int(settings["seed"]))
	eager = create_coverage_model(settings).to(device)
	model: nn.Module = eager
	optimizer = torch.optim.AdamW(eager.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]), fused=device.type == "cuda")
	scaler = GradScaler("cuda", enabled=precision == "fp16")
	if settings["class_weighting"] not in (False, True, "sqrt_play"):
		raise ValueError("Class weighting must be false, true or sqrt_play")
	weights = None
	if settings["class_weighting"]:
		counts = torch.bincount(y_train[torch.from_numpy((~train_index.augmented).to_numpy())], minlength=int(settings["num_classes"])).float()
		weights = (counts.sum() / (len(counts) * counts.clamp_min(1))).clamp_max(5).to(device)
		if settings["class_weighting"] == "sqrt_play":
			unique = train_index.loc[~train_index.augmented].drop_duplicates(["season", "gameId", "playId"]).index.to_numpy()
			counts = torch.bincount(y_train[torch.from_numpy(unique)], minlength=int(settings["num_classes"])).float()
			weights = (counts.sum() / (len(counts) * counts.clamp_min(1))).sqrt().clamp_max(3).to(device)
		settings["fitted_class_weights"] = weights.cpu().tolist()
	loss_fn = nn.CrossEntropyLoss(weight=weights)
	val_loader = DataLoader(TensorDataset(x_val, mask_val, y_val), batch_size=batch_size, pin_memory=device.type == "cuda", num_workers=0)
	pipeline = str(settings["input_pipeline"])
	if pipeline not in ("loader", "direct", "device"):
		raise ValueError("Unknown input pipeline")
	if pipeline == "device":
		if device.type != "cuda":
			raise ValueError("Device-resident input requires CUDA")
		x_train, mask_train, y_train = x_train.to(device), mask_train.to(device), y_train.to(device)
	elif pipeline == "direct" and device.type == "cuda":
		x_train, mask_train, y_train = x_train.pin_memory(), mask_train.pin_memory(), y_train.pin_memory()
	history: list[dict[str, Any]] = []
	start_epoch, elapsed_base, best = 0, 0.0, (-1.0, -math.inf)
	bad_epochs = 0
	compile_fallback = None
	if resume:
		state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
		if state["identity"] != identity or state["dataset_hash"] != manifest["dataset_hash"]:
			raise ValueError("Resume config or dataset mismatch")
		eager.load_state_dict(state["model"])
		optimizer.load_state_dict(state["optimizer"])
		scaler.load_state_dict(state["scaler"])
		generator.set_state(state["generator"])
		torch.set_rng_state(state["rng"])
		if device.type == "cuda":
			torch.cuda.set_rng_state_all(state["cuda_rng"])
		start_epoch, history = state["epoch"] + 1, state["history"]
		best, bad_epochs, elapsed_base = tuple(state["best"]), state["bad_epochs"], state["elapsed_seconds"]
	if settings["compile"]:
		os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(run_dir / "compiler-cache"))
		original = {key: value.detach().clone() for key, value in eager.state_dict().items()}
		rng = torch.get_rng_state()
		cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
		try:
			model = cast(nn.Module, torch.compile(eager))
			with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
				forward_model(model, x_train[:batch_size].to(device), mask_train[:batch_size].to(device)).sum().backward()
		except (RuntimeError, ValueError) as exc:
			model, compile_fallback = eager, str(exc)
		finally:
			eager.load_state_dict(original)
			eager.zero_grad(set_to_none=True)
			torch.set_rng_state(rng)
			if device.type == "cuda":
				torch.cuda.set_rng_state_all(cuda_rng)
	(run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
	try:
		for epoch in range(start_epoch, epochs):
			if int(settings["patience"]) > 0 and bad_epochs >= int(settings["patience"]):
				break
			epoch_start = time.monotonic()
			eager.train()
			loss_sum = torch.zeros((), device=device)
			order = torch.randperm(len(x_train), generator=generator) if sampling_weights is None else torch.multinomial(sampling_weights, len(x_train), replacement=True, generator=generator)
			if pipeline == "loader":
				batches = iter(
					DataLoader(
						TensorDataset(x_train, mask_train, y_train),
						batch_size=batch_size,
						sampler=order.tolist(),
						generator=torch.Generator().manual_seed(0),
						pin_memory=device.type == "cuda",
						num_workers=0,
					)
				)
			else:
				order = order.to(x_train.device)
				batches = ((x_train[ids], mask_train[ids], y_train[ids]) for ids in order.split(batch_size))
			for features, time_mask, targets in batches:
				features, time_mask, targets = features.to(device, non_blocking=True), time_mask.to(device, non_blocking=True), targets.to(device, non_blocking=True)
				optimizer.zero_grad(set_to_none=True)
				with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
					loss = loss_fn(forward_model(model, features, time_mask), targets)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				loss_sum += loss.detach() * len(features)
			train_loss = float(loss_sum.cpu()) / len(x_train)
			eager.eval()
			outputs = []
			with torch.inference_mode():
				for features, time_mask, _ in val_loader:
					outputs.append(forward_model(model, features.to(device, non_blocking=True), time_mask.to(device, non_blocking=True)).float())
			metrics = _score(torch.cat(outputs), y_val, int(settings["num_classes"]))
			if not math.isfinite(train_loss) or not math.isfinite(metrics["log_loss"]):
				raise ValueError("Nonfinite training or validation loss")
			row = {"epoch": epoch, "train_loss": train_loss, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"], "log_loss": metrics["log_loss"]}
			history.append(row)
			quality = (metrics["macro_f1"], -metrics["log_loss"])
			if quality > best:
				best, bad_epochs = quality, 0
				save_bundle(eager, run_dir / "bundle", settings, manifest["dataset_hash"], epoch)
			else:
				bad_epochs += 1
			elapsed = elapsed_base + time.monotonic() - started
			state = {
				"epoch": epoch,
				"model": eager.state_dict(),
				"optimizer": optimizer.state_dict(),
				"scaler": scaler.state_dict(),
				"generator": generator.get_state(),
				"rng": torch.get_rng_state(),
				"cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
				"history": history,
				"best": list(best),
				"bad_epochs": bad_epochs,
				"elapsed_seconds": elapsed,
				"identity": identity,
				"dataset_hash": manifest["dataset_hash"],
			}
			torch.save(state, run_dir / "latest.tmp")
			(run_dir / "latest.tmp").replace(run_dir / "latest.pt")
			with (run_dir / "epochs.jsonl").open("a") as stream:
				stream.write(json.dumps({**row, "epoch_seconds": time.monotonic() - epoch_start}) + "\n")
			if reporter is not None:
				reporter(row, run_dir)
			if int(settings["patience"]) > 0 and bad_epochs >= int(settings["patience"]):
				break
		if not (run_dir / "bundle/metadata.json").exists():
			raise ValueError("Training produced no model bundle")
		receipt = {
			"status": "complete",
			"device": str(device),
			"precision": precision,
			"epochs_completed": len(history),
			"fit_examples": len(x_train),
			"fit_plays": len(train_index.drop_duplicates(["season", "gameId", "playId"])),
			"optimizer_steps": len(history) * math.ceil(len(x_train) / batch_size),
			"elapsed_seconds": elapsed_base + time.monotonic() - started,
			"best_macro_f1": best[0],
			"compile_fallback": compile_fallback,
			"dataset_hash": manifest["dataset_hash"],
			"manifest_sha256": file_hash(manifest_path),
		}
	except BaseException as exc:
		(run_dir / "terminal.json").write_text(json.dumps({"status": "failed", "reason": str(exc), "device": str(device)}) + "\n")
		raise
	(run_dir / "terminal.json").write_text(json.dumps(receipt, indent=2) + "\n")
	return run_dir / "bundle"
