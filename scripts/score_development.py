"""Fit matched CPU comparators or score completed pilot bundles on outer folds."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from dataset import file_hash
from dataset_io import load_examples
from evaluate import KEYS, PROB_COLUMNS, score_predictions
from generate_predictions import predict_batches
from model_bundle import load_bundle
from models.baselines import fit_baseline, predict_baseline_proba


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("root", type=Path)
	parser.add_argument("--baselines", action="store_true")
	parser.add_argument("--batch", action="store_true", help="Score run_development_batch output instead of the pilot")
	args = parser.parse_args()
	torch.set_num_threads(4)
	if args.batch:
		jobs = json.loads((args.root / "plan.json").read_text())
		folds = [(str(number), Path(job["manifest"]), args.root / f"job-{number:02d}", len(job["trials"])) for number, job in enumerate(jobs)]
	else:
		folds = [(str(week), args.root / f"fold-{week}/manifest.json", args.root / f"pilot-{week}", 3) for week in (3, 4)]
	for week, manifest, run, count in folds:
		out = args.root / f"scores-{week}"
		out.mkdir(exist_ok=True)
		models = {}
		if args.baselines:
			x, _, y, _ = load_examples(manifest, "fit")
			x = x[:, -1].contiguous().numpy()
			for name in ("logistic", "lightgbm"):
				started = time.monotonic()
				model = fit_baseline(x, y.numpy(), name)
				joblib.dump(model, out / f"{name}.joblib")
				models[name] = (model, time.monotonic() - started)
			del x, y
		else:
			for trial in range(count):
				bundle = run / f"trial-{trial:03d}/bundle"
				if not (bundle.parent / "terminal.json").exists() or json.loads((bundle.parent / "terminal.json").read_text())["status"] != "complete":
					continue
				model, _ = load_bundle(bundle, torch.device("cpu"))
				models[f"neural-{trial}"] = (model, None)
		for offset in (-1, -10):
			x, mask, y, index = load_examples(manifest, "development", offset=offset)
			for name, (model, seconds) in models.items():
				started = time.monotonic()
				probs = predict_baseline_proba(model, x[:, -1].numpy()) if args.baselines else predict_batches(model, x, mask, 512, torch.device("cpu")).numpy()
				inference_seconds = time.monotonic() - started
				predictions = index[[*KEYS, "target"]].copy()
				predictions[PROB_COLUMNS] = probs
				path = out / f"{name}_{offset}.parquet"
				predictions.to_parquet(path, index=False)
				metrics = score_predictions(y.numpy(), probs)
				metrics.update(
					macro_precision=float(np.mean([v["precision"] for v in metrics["per_class"].values()])), macro_recall=float(np.mean([v["recall"] for v in metrics["per_class"].values()]))
				)
				metrics.update(manifest_sha256=file_hash(manifest), predictions_sha256=file_hash(path), fit_seconds=seconds, cpu_inference_seconds=inference_seconds, games=int(index.gameId.nunique()))
				path.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
				print(week, name, offset, metrics["macro_f1"], flush=True)


if __name__ == "__main__":
	main()
