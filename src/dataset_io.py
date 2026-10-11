"""Verified reads of frozen datasets, with a separate final-evaluation gate."""

import json
from pathlib import Path

import pandas as pd
import torch

from coverage_labels import CLASSES
from dataset import FEATURES, file_hash


def load_examples(manifest_path: Path, split: str, offset: int | None = None, season: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
	manifest = json.loads(manifest_path.read_text())
	if manifest["classes"] != list(CLASSES) or manifest["features"] != list(FEATURES) or manifest.get("status") != "complete":
		raise ValueError("Incomplete or incompatible dataset manifest")
	if split not in ("fit", "tune", "calibrate", "final", "historical"):
		raise ValueError("Unknown dataset split")
	if split == "final":
		receipt_path = manifest_path.parent / "selection.json"
		if not receipt_path.exists():
			raise ValueError("Final data requires a frozen selection receipt")
		receipt = json.loads(receipt_path.read_text())
		if not receipt.get("frozen") or receipt.get("dataset_hash") != manifest["dataset_hash"]:
			raise ValueError("Invalid selection receipt")
	index_path = manifest_path.parent / "examples.parquet"
	if file_hash(index_path) != manifest["examples_sha256"]:
		raise ValueError("Dataset index hash mismatch")
	index = pd.read_parquet(index_path)
	selected = index.split.eq(split)
	if offset is not None:
		selected &= index.offset.eq(offset)
	if season is not None:
		selected &= index.season.eq(season)
	index = index.loc[selected].copy()
	if index.empty:
		raise ValueError(f"No eligible examples for {split}, offset={offset}, season={season}")
	x_parts, mask_parts, y_parts, indexes = [], [], [], []
	shards = {entry["path"]: entry for entry in manifest["shards"]}
	for name, rows in index.groupby("shard", sort=True):
		path = manifest_path.parent / str(name)
		entry = shards[str(name)]
		if file_hash(path) != entry["sha256"]:
			raise ValueError(f"Shard hash mismatch: {name}")
		shard = torch.load(path, map_location="cpu", weights_only=True)
		x, mask, y = shard["features"], shard["time_mask"], shard["targets"]
		if x.shape != (entry["rows"], manifest["settings"]["history_frames"], 22, 5) or mask.shape != x.shape[:2] or y.shape != (len(x),):
			raise ValueError("Shard shape mismatch")
		if x.dtype != torch.float32 or mask.dtype != torch.bool or y.dtype != torch.long or not torch.isfinite(x).all() or not mask[:, -1].all():
			raise ValueError("Invalid shard values or types")
		ids = torch.from_numpy(rows.shard_row.to_numpy(dtype="int64"))
		if not torch.equal(y[ids], torch.from_numpy(rows.target.to_numpy(dtype="int64"))):
			raise ValueError("Index/shard target mismatch")
		x_parts.append(x[ids])
		mask_parts.append(mask[ids])
		y_parts.append(y[ids])
		indexes.append(rows)
	return torch.cat(x_parts), torch.cat(mask_parts), torch.cat(y_parts), pd.concat(indexes, ignore_index=True)
