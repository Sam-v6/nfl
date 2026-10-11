"""Batched, keyed prediction using verified causal inputs and portable bundles."""

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dataset import file_hash
from dataset_io import load_examples
from evaluate import KEYS
from model_bundle import load_bundle
from models.factory import forward_model
from training import select_device


def predict_batches(model: nn.Module, features: torch.Tensor, time_mask: torch.Tensor, batch_size: int, device: torch.device, precision: str = "fp32", temperature: float = 1.0) -> torch.Tensor:
	if batch_size < 1 or not len(features) or temperature <= 0:
		raise ValueError("Require positive batch size, nonempty input and positive temperature")
	if precision not in ("fp32", "bf16", "fp16") or (device.type == "cpu" and precision != "fp32"):
		raise ValueError("Unsupported prediction precision")
	if precision == "bf16" and not torch.cuda.is_bf16_supported():
		raise ValueError("BF16 is unavailable")
	dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
	outputs = []
	model.eval()
	with torch.inference_mode():
		for start in range(0, len(features), batch_size):
			x = features[start : start + batch_size].to(device, non_blocking=True)
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
				logits = forward_model(model, x, time_mask[start : start + batch_size].to(device, non_blocking=True))
			outputs.append((logits.float() / temperature).softmax(-1).cpu())
	return torch.cat(outputs)


def predict_dataset(
	bundle_path: Path,
	manifest_path: Path,
	output_dir: Path,
	batch_size: int,
	split: str = "tune",
	device: str = "auto",
	precision: str = "fp32",
	offset: int | None = None,
) -> Path:
	if output_dir.exists():
		raise ValueError("Prediction output already exists")
	started = time.monotonic()
	selected_device = select_device(device)
	model, metadata = load_bundle(bundle_path, selected_device)
	x, mask, _, index = load_examples(manifest_path, split, offset=offset, require_targets=False)
	if "augmented" in index:
		keep = ~index.augmented
		x, mask = x[torch.from_numpy(keep.to_numpy())], mask[torch.from_numpy(keep.to_numpy())]
		index = index.loc[keep].copy()
	if index.duplicated(KEYS).any():
		raise ValueError("Duplicate prediction keys")
	probabilities = predict_batches(model, x, mask, batch_size, selected_device, precision, float(metadata.get("temperature", 1.0)))
	output = index[[*KEYS, "offset", "split"]].reset_index(drop=True)
	for i in range(len(metadata["classes"])):
		output[f"prob_{i}"] = probabilities[:, i].numpy()
	output_dir.mkdir(parents=True)
	path = output_dir / "predictions.parquet"
	output.to_parquet(path, index=False)
	receipt: dict[str, Any] = {
		"status": "complete",
		"rows": len(output),
		"device": str(selected_device),
		"precision": precision,
		"batch_size": batch_size,
		"classes": metadata["classes"],
		"elapsed_seconds": time.monotonic() - started,
		"manifest_sha256": file_hash(manifest_path),
		"weights_sha256": metadata["weights_sha256"],
		"predictions_sha256": file_hash(path),
	}
	(output_dir / "terminal.json").write_text(json.dumps(receipt, indent=2) + "\n")
	return path
