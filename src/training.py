"""One train/evaluate loop for local runs and Ray trials, with portable resume."""

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from dataset import file_hash
from dataset_io import load_examples
from evaluate import _macro_from_cm, score_predictions
from model_bundle import save_bundle
from models.transformer import create_transformer_model

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
}


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
	identity = {key: value for key, value in settings.items() if key != "epochs"}
	x_train, _, y_train, _ = load_examples(manifest_path, "fit")
	x_val, _, y_val, _ = load_examples(manifest_path, "tune", offset=-1)
	x_train, x_val = x_train[:, -1].contiguous(), x_val[:, -1].contiguous()
	if settings["num_classes"] == 2:
		y_train = ((y_train == 0) | (y_train == 1) | (y_train == 5)).long()
		y_val = ((y_val == 0) | (y_val == 1) | (y_val == 5)).long()
	manifest = json.loads(manifest_path.read_text())
	run_dir.mkdir(parents=True, exist_ok=resume)
	started = time.monotonic()
	torch.manual_seed(int(settings["seed"]))
	if device.type == "cuda":
		torch.cuda.manual_seed_all(int(settings["seed"]))
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
	generator = torch.Generator().manual_seed(int(settings["seed"]))
	eager = create_transformer_model(settings).to(device)
	model = eager
	optimizer = torch.optim.AdamW(eager.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]), fused=device.type == "cuda")
	scaler = GradScaler("cuda", enabled=precision == "fp16")
	loss_fn = nn.CrossEntropyLoss()
	train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=generator, pin_memory=device.type == "cuda", num_workers=0)
	val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, pin_memory=device.type == "cuda", num_workers=0)
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
		original = {key: value.detach().clone() for key, value in eager.state_dict().items()}
		rng = torch.get_rng_state()
		cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
		try:
			model = torch.compile(eager)
			with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
				model(x_train[:batch_size].to(device)).sum().backward()
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
			epoch_start = time.monotonic()
			eager.train()
			loss_sum = torch.zeros((), device=device)
			for features, targets in train_loader:
				features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
				optimizer.zero_grad(set_to_none=True)
				with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
					loss = loss_fn(model(features), targets)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				loss_sum += loss.detach() * len(features)
			train_loss = float(loss_sum.cpu()) / len(x_train)
			eager.eval()
			outputs = []
			with torch.inference_mode():
				for features, _ in val_loader:
					with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
						outputs.append(model(features.to(device, non_blocking=True)).float())
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
