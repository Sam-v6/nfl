import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_model_bundle import MODEL_CONFIG
from test_training import training_fixture


def test_prediction_batches_preserve_keys_tail_and_reference(tmp_path: Path) -> None:
	from dataset import file_hash
	from generate_predictions import predict_dataset
	from model_bundle import save_bundle
	from models.transformer import create_transformer_model

	manifest = training_fixture(tmp_path / "data")
	index_path = manifest.parent / "examples.parquet"
	index = pd.read_parquet(index_path).sample(frac=1, random_state=17).reset_index(drop=True)
	index.to_parquet(index_path, index=False)
	info = json.loads(manifest.read_text())
	info["examples_sha256"] = file_hash(index_path)
	manifest.write_text(json.dumps(info))
	model = create_transformer_model(MODEL_CONFIG).eval()
	save_bundle(model, tmp_path / "bundle", MODEL_CONFIG, "fixture")
	shard = torch.load(manifest.parent / "tune.pt", weights_only=True)
	with torch.inference_mode():
		reference = model(shard["features"][:, -1]).softmax(-1)
	for batch_size in (1, 7, 64):
		path = predict_dataset(tmp_path / "bundle", manifest, tmp_path / str(batch_size), batch_size, split="tune", device="cpu")
		predictions = pd.read_parquet(path).sort_values("playId")
		assert len(predictions) == 16
		assert not predictions.duplicated(["season", "gameId", "playId", "cutoff_frame"]).any()
		torch.testing.assert_close(torch.tensor(predictions[[f"prob_{i}" for i in range(7)]].to_numpy()), reference, atol=1e-5, rtol=1e-4)
	with pytest.raises(ValueError, match="exists"):
		predict_dataset(tmp_path / "bundle", manifest, tmp_path / "7", 7, split="tune", device="cpu")


def test_unlabeled_dataset_prediction_needs_no_targets(tmp_path: Path) -> None:
	from dataset import file_hash
	from generate_predictions import predict_dataset
	from model_bundle import save_bundle
	from models.transformer import create_transformer_model

	manifest = training_fixture(tmp_path / "data")
	index_path = manifest.parent / "examples.parquet"
	pd.read_parquet(index_path).drop(columns=["target"]).to_parquet(index_path, index=False)
	info = json.loads(manifest.read_text())
	info["examples_sha256"] = file_hash(index_path)
	for entry in info["shards"]:
		path = manifest.parent / entry["path"]
		data = torch.load(path, weights_only=True)
		data.pop("targets")
		torch.save(data, path)
		entry["sha256"] = file_hash(path)
	manifest.write_text(json.dumps(info))
	save_bundle(create_transformer_model(MODEL_CONFIG), tmp_path / "bundle", MODEL_CONFIG, "fixture")
	result = predict_dataset(tmp_path / "bundle", manifest, tmp_path / "out", 7, split="tune", device="cpu")
	assert "target" not in pd.read_parquet(result)
	assert len(pd.read_parquet(result)) == 16


def test_raw_prefix_inference_is_unlabeled_and_rejects_observed_snap(tmp_path: Path) -> None:
	from test_dataset import frames

	from inference_data import prepare_inference_dataset

	tracking = frames().assign(gameId=1, playId=2)
	requests = pd.DataFrame([{"season": 2022, "gameId": 1, "playId": 2, "cutoff_frame": 100, "defensiveTeam": "DEF"}])
	manifest = prepare_inference_dataset(tracking, requests, tmp_path / "inputs")
	assert not json.loads(manifest.read_text())["labeled"]
	assert "target" not in pd.read_parquet(manifest.parent / "examples.parquet")
	tracking["event"] = "ball_snap"
	with pytest.raises(ValueError, match="snap"):
		prepare_inference_dataset(tracking, requests, tmp_path / "rejected")
	assert not (tmp_path / "rejected").exists()
