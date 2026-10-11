"""Bounded real-data verification, restricted to fit/tune/calibrate roles."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from audit_data import audit_data
from calibrate import calibrate_bundle
from dataset import file_hash, prepare_dataset
from evaluate import evaluate_predictions
from experiment import RunLock
from generate_predictions import predict_dataset
from model_bundle import load_bundle
from training import train_model


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-root", type=Path, required=True)
	parser.add_argument("--run-dir", type=Path, required=True)
	parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
	args = parser.parse_args()
	if args.run_dir.exists():
		parser.error("Integration output already exists")
	args.run_dir.mkdir(parents=True)
	audit_data(args.data_root, args.run_dir / "audit")
	# A real-data smoke check needs a bounded subset, not a second full preparation.
	import pandas as pd
	import polars as pl

	from coverage_labels import canonicalize_coverage

	inputs = args.run_dir / "inputs"
	base = inputs / "2022"
	base.mkdir(parents=True)
	games = pd.read_parquet(args.data_root / "2022/games.parquet")
	plays = pd.read_parquet(args.data_root / "2022/plays.parquet").merge(games[["gameId", "week"]], on="gameId", validate="many_to_one")
	eligible = plays.passResult.isin(["C", "I", "IN", "S"]) & plays.pff_passCoverage.map(canonicalize_coverage).notna()
	selected = plays.loc[eligible & plays.week.isin([1, 5, 6])].groupby("week", sort=True).head(28)
	games.loc[games.gameId.isin(selected.gameId)].to_parquet(base / "games.parquet", index=False)
	selected.drop(columns="week").to_parquet(base / "plays.parquet", index=False)
	for week in (1, 5, 6):
		keys = pl.from_pandas(selected.loc[selected.week.eq(week), ["gameId", "playId"]]).lazy()
		pl.scan_parquet(args.data_root / f"2022/tracking_week_{week}.parquet").join(keys, on=["gameId", "playId"], how="semi").collect().write_parquet(base / f"tracking_week_{week}.parquet")
	(args.run_dir / "input-subset.json").write_text(
		json.dumps(
			{
				"kind": "bounded real-data integration subset",
				"source_sha256": {str(p.relative_to(args.data_root)): file_hash(p) for p in (args.data_root / "2022").glob("*.parquet")},
				"plays_per_week_limit": 28,
			},
			indent=2,
		)
		+ "\n"
	)
	manifest = prepare_dataset(inputs, {"seasons": [2022], "cutoffs": [-10, -1], "history_frames": 10, "seed": 42}, args.run_dir / "dataset")
	settings = {"model_dim": 16, "num_heads": 2, "num_layers": 1, "device": args.device, "epochs": 2, "batch_size": 512, "cpu_threads": 2, "precision": "fp32"}
	import os

	with RunLock(Path(f"/tmp/nfl-gpu-{os.getuid()}.lock")) if args.device == "cuda" else nullcontext():
		train_model(settings, manifest, args.run_dir / "training")
		bundle = train_model({**settings, "resume": True, "epochs": 3}, manifest, args.run_dir / "training")
		load_bundle(bundle, torch.device("cpu"))
		checkpoint = torch.load(args.run_dir / "training/latest.pt", weights_only=True)
		assert checkpoint["epoch"] == 2
		calibrated = calibrate_bundle(bundle, manifest, args.run_dir / "calibrated", device=args.device)
		predictions = predict_dataset(calibrated, manifest, args.run_dir / "predictions", 512, split="tune", device=args.device)
	metrics = evaluate_predictions(pd.read_parquet(predictions), manifest, args.run_dir / "report")
	(args.run_dir / "terminal.json").write_text(
		json.dumps({"status": "complete", "device": args.device, "epochs_completed": 3, "metrics": str(metrics), "final_features_read": False}, indent=2) + "\n"
	)


if __name__ == "__main__":
	main()
