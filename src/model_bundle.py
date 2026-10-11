"""Portable eager weights with explicit vocabulary, schema and provenance."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from coverage_labels import CLASSES
from dataset import FEATURES, file_hash
from models.transformer import create_transformer_model


def save_bundle(model: nn.Module, path: Path, config: dict[str, Any], dataset_hash: str, best_epoch: int | None = None) -> None:
	path.mkdir(parents=True, exist_ok=True)
	eager = getattr(model, "_orig_mod", model)
	state = {key: value.detach().cpu().clone() for key, value in eager.state_dict().items()}
	temporary = path / "weights.tmp"
	torch.save(state, temporary)
	temporary.replace(path / "weights.pt")
	classes = list(CLASSES) if int(config.get("num_classes", 7)) == 7 else ["Zone", "Man"]
	metadata = {
		"format_version": 1,
		"classes": classes,
		"features": list(FEATURES),
		"config": config,
		"dataset_hash": dataset_hash,
		"weights_sha256": file_hash(path / "weights.pt"),
		"best_epoch": best_epoch,
	}
	info = path / "metadata.tmp"
	info.write_text(json.dumps(metadata, indent=2) + "\n")
	info.replace(path / "metadata.json")


def load_bundle(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
	metadata = json.loads((path / "metadata.json").read_text())
	if metadata["classes"] not in (list(CLASSES), ["Zone", "Man"]) or metadata["features"] != list(FEATURES):
		raise ValueError("Incompatible class vocabulary or feature schema")
	if file_hash(path / "weights.pt") != metadata["weights_sha256"]:
		raise ValueError("Model weights hash mismatch")
	config = metadata["config"]
	if int(config.get("num_classes", 2)) != len(metadata["classes"]) or int(config.get("feature_len", 5)) != 5:
		raise ValueError("Model dimensions disagree with metadata")
	model = create_transformer_model(config).to(device)
	state = torch.load(path / "weights.pt", map_location=device, weights_only=True)
	model.load_state_dict({name.removeprefix("_orig_mod."): tensor for name, tensor in state.items()})
	return model.eval(), metadata


def predict_proba(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
	return torch.softmax(model(features).float(), dim=-1)


def import_legacy_binary(weights_path: Path, config: dict[str, Any], output_dir: Path) -> Path:
	"""Convert explicitly described binary weights; never infer labels from a filename."""
	if config.get("num_classes") != 2 or config.get("feature_len") != 5:
		raise ValueError("Legacy conversion requires explicit num_classes=2 and feature_len=5")
	if output_dir.exists():
		raise ValueError("Legacy bundle output already exists")
	model = create_transformer_model(config)
	state = torch.load(weights_path, map_location="cpu", weights_only=True)
	model.load_state_dict({name.removeprefix("_orig_mod."): tensor for name, tensor in state.items()})
	save_bundle(model, output_dir, config, "legacy-unverified:" + file_hash(weights_path))
	return output_dir
