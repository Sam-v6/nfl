"""Prepare unlabeled prediction requests from observation prefixes, without outcomes."""

import json
from pathlib import Path

import pandas as pd
import torch

from coverage_labels import CLASSES
from dataset import FEATURES, file_hash, prepare_prefix
from evaluate import KEYS


def prepare_inference_dataset(tracking: pd.DataFrame, requests: pd.DataFrame, output_dir: Path, history_frames: int = 30) -> Path:
	if output_dir.exists():
		raise ValueError("Inference dataset output already exists")
	if requests.empty or requests.duplicated(KEYS).any() or not 1 <= history_frames <= 150:
		raise ValueError("Require unique nonempty requests and 1–150 history frames")
	groups = {(int(rows.iloc[0].gameId), int(rows.iloc[0].playId)): rows for _, rows in tracking.groupby(["gameId", "playId"], sort=False)}
	features, masks, records = [], [], []
	for request in requests.to_dict("records"):
		prefix = groups[(int(request["gameId"]), int(request["playId"]))]
		cutoff = int(request["cutoff_frame"])
		prefix = prefix.loc[prefix.frameId.le(cutoff)]
		if "event" in prefix and prefix.event.eq("ball_snap").any():
			raise ValueError("Prediction cutoff must precede any observed snap")
		x, mask = prepare_prefix(prefix, cutoff, str(request["defensiveTeam"]), history_frames)
		features.append(torch.from_numpy(x))
		masks.append(torch.from_numpy(mask))
		records.append({**{key: request[key] for key in KEYS}, "offset": request.get("offset", 0), "split": "predict", "augmented": False, "shard": "inputs.pt", "shard_row": len(records)})
	output_dir.mkdir(parents=True)
	torch.save({"features": torch.stack(features), "time_mask": torch.stack(masks)}, output_dir / "inputs.pt")
	pd.DataFrame(records).to_parquet(output_dir / "examples.parquet", index=False)
	input_hash = file_hash(output_dir / "inputs.pt")
	manifest = {
		"status": "complete",
		"classes": list(CLASSES),
		"features": list(FEATURES),
		"settings": {"history_frames": history_frames},
		"dataset_hash": input_hash,
		"examples_sha256": file_hash(output_dir / "examples.parquet"),
		"shards": [{"path": "inputs.pt", "rows": len(records), "sha256": input_hash}],
		"labeled": False,
	}
	path = output_dir / "manifest.json"
	path.write_text(json.dumps(manifest, indent=2) + "\n")
	return path
