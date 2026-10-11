"""Explicit, isolated entry points for coverage experiments."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from common.paths import RunPaths


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(prog="nfl", description="Reproducible NFL coverage experiments")
	commands = parser.add_subparsers(dest="command", required=True)
	for command in ("audit", "prepare", "train", "predict", "evaluate", "benchmark", "experiment"):
		sub = commands.add_parser(command)
		sub.add_argument("--data-root", type=Path, required=True)
		sub.add_argument("--run-dir", type=Path, required=True)
		sub.add_argument("--config", type=Path)
		sub.add_argument("--predictions", type=Path)
		sub.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
	args = parser.parse_args(argv)
	paths = RunPaths.resolve(args.data_root, args.run_dir)
	try:
		if args.command == "audit":
			from audit_data import audit_data

			print(audit_data(paths.data_root, paths.run_dir))
		elif args.command == "prepare":
			from dataset import prepare_dataset

			logging.basicConfig(level=logging.INFO)

			config = json.loads(args.config.read_text()) if args.config else {}
			print(prepare_dataset(paths.data_root, config, paths.run_dir))
		elif args.command == "evaluate":
			if args.predictions is None:
				parser.error("evaluate requires --predictions")
			import pandas as pd

			from evaluate import evaluate_predictions

			print(evaluate_predictions(pd.read_parquet(args.predictions), paths.data_root / "manifest.json", paths.run_dir))
		elif args.command == "train":
			config = json.loads(args.config.read_text()) if args.config else {}
			if args.device != "auto":
				config["device"] = args.device
			if config.get("model_type") == "baselines":
				from models.baselines import train_baselines

				print(train_baselines(paths.data_root / "manifest.json", paths.run_dir))
			else:
				from training import train_model

				print(train_model(config, paths.data_root / "manifest.json", paths.run_dir))
		elif args.command == "predict":
			from generate_predictions import predict_dataset

			config = json.loads(args.config.read_text()) if args.config else {}
			if "bundle" not in config:
				parser.error("predict requires a config naming the model bundle")
			print(
				predict_dataset(
					Path(config["bundle"]),
					paths.data_root / "manifest.json",
					paths.run_dir,
					int(config.get("batch_size", 512)),
					split=config.get("split", "predict"),
					device=args.device if args.device != "auto" else config.get("device", "auto"),
					precision=config.get("precision", "fp32"),
					offset=config.get("offset"),
				)
			)
		elif args.command == "benchmark":
			from benchmark import benchmark

			config = json.loads(args.config.read_text()) if args.config else {}
			if "bundle" not in config:
				parser.error("benchmark requires a config naming the model bundle")
			config["manifest"] = str(paths.data_root / "manifest.json")
			if args.device != "auto":
				config["device"] = args.device
			print(benchmark(config, paths.run_dir))
		else:
			parser.error(f"{args.command} is not implemented yet")
	except (OSError, ValueError) as exc:
		parser.error(str(exc))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
