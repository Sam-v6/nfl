"""Matched CPU screens for permitted context and observed player motion."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier

from dataset import file_hash
from dataset_io import load_examples
from development_features import motion_summary
from evaluate import KEYS, PROB_COLUMNS, score_predictions
from models.baselines import geometric_summary


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("root", type=Path)
	parser.add_argument("raw", type=Path)
	parser.add_argument("output", type=Path)
	args = parser.parse_args()
	args.output.mkdir(parents=True, exist_ok=False)
	torch.set_num_threads(4)
	context = pd.concat([pd.read_parquet(args.raw / str(year) / "plays.parquet", columns=["gameId", "playId", "down", "yardsToGo"]).assign(season=year) for year in (2021, 2022)])
	if context.duplicated(["season", "gameId", "playId"]).any() or context[["down", "yardsToGo"]].isna().any().any():
		raise ValueError("Invalid context join")
	report = {"context_columns": ["down", "yardsToGo"], "raw_hashes": {str(year): file_hash(args.raw / str(year) / "plays.parquet") for year in (2021, 2022)}, "scores": []}
	for week in (3, 4):
		manifest = args.root / f"fold-{week}/manifest.json"
		for hypothesis in ("context", "motion"):
			started = time.monotonic()
			x, mask, y, index = load_examples(manifest, "fit")
			keep = np.ones(len(index), dtype=bool) if hypothesis == "context" else (index.season.eq(2022).to_numpy() & (mask.sum(1).numpy() >= 10))
			x, mask, y, index = x[keep], mask[keep], y[keep], index.loc[keep]
			geometry = geometric_summary(x[:, -1].numpy())
			extra = (
				index.merge(context, on=["season", "gameId", "playId"], validate="many_to_one", how="left")[["down", "yardsToGo"]].to_numpy()
				if hypothesis == "context"
				else motion_summary(x, mask).numpy()
			)
			if not np.isfinite(extra).all():
				raise ValueError("Missing joined context")
			models = []
			for features in (geometry, np.concatenate([geometry, extra], axis=1)):
				model = LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, n_jobs=4, random_state=42, verbosity=-1)
				models.append(model.fit(pd.DataFrame(features), y.numpy()))
			fit_plays = len(index.drop_duplicates(["season", "gameId", "playId"]))
			del x, mask, y, index, geometry, extra
			for offset in (-1, -10):
				x, mask, y, index = load_examples(manifest, "development", offset=offset)
				keep = np.ones(len(index), dtype=bool) if hypothesis == "context" else (index.season.eq(2022).to_numpy() & (mask.sum(1).numpy() >= 10))
				x, mask, y, index = x[keep], mask[keep], y[keep], index.loc[keep]
				geometry = geometric_summary(x[:, -1].numpy())
				extra = (
					index.merge(context, on=["season", "gameId", "playId"], validate="many_to_one", how="left")[["down", "yardsToGo"]].to_numpy()
					if hypothesis == "context"
					else motion_summary(x, mask).numpy()
				)
				for enriched, (model, features) in enumerate(zip(models, (geometry, np.concatenate([geometry, extra], axis=1)), strict=True)):
					probs = np.zeros((len(y), 7))
					probs[:, model.classes_.astype(int)] = model.predict_proba(pd.DataFrame(features))
					predictions = index[[*KEYS, "target"]].copy()
					predictions[PROB_COLUMNS] = probs
					predictions.to_parquet(args.output / f"{week}-{hypothesis}-{enriched}-{offset}.parquet", index=False)
					metrics = score_predictions(y.numpy(), probs)
					metrics.update(week=week, hypothesis=hypothesis, enriched=bool(enriched), offset=offset, fit_plays=fit_plays, games=int(index.gameId.nunique()))
					report["scores"].append(metrics)
					print(week, hypothesis, enriched, offset, metrics["macro_f1"], flush=True)
			report[f"seconds_{week}_{hypothesis}"] = time.monotonic() - started
			(args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
	main()
