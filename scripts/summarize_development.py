"""Summarize the predeclared development campaign without promoting a model."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from evaluate import PROB_COLUMNS, compare_predictions, score_predictions

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("root", type=Path)
	parser.add_argument("output", type=Path)
	args = parser.parse_args()
	args.output.mkdir(parents=True, exist_ok=True)
	root = args.root
	arms = {
		"Reference: 30 epochs": ("", "neural-0"),
		"Up to 100 epochs": ("", "neural-1"),
		"Equal-play sampling": ("", "neural-2"),
		"Mild class weighting": ("objective/", "neural-0"),
		"LightGBM": ("", "lightgbm"),
		"Logistic": ("", "logistic"),
	}
	summary = {"evidence": "historically exposed chronological development; one seed", "arms": {}, "learning_curves": {}, "compute": json.loads((root / "compute-receipt.json").read_text())}
	for name, (prefix, model) in arms.items():
		folds = [json.loads((root / f"{prefix}scores-{i if prefix else w}/{model}_-1.json").read_text()) for i, w in enumerate((3, 4))]
		metrics = {key: float(np.mean([fold[key] for fold in folds])) for key in ("macro_f1", "macro_precision", "macro_recall", "accuracy", "log_loss")}
		metrics["binary_f1"] = float(np.mean([fold["binary"]["macro_f1"] for fold in folds]))
		metrics["one_second_f1"] = float(np.mean([json.loads((root / f"{prefix}scores-{i if prefix else w}/{model}_-10.json").read_text())["macro_f1"] for i, w in enumerate((3, 4))]))
		metrics["folds"] = folds
		summary["arms"][name] = metrics
	fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
	for i, week in enumerate((3, 4)):
		for model, label in (("neural-0", "Neural"), ("lightgbm", "LightGBM"), ("logistic", "Logistic")):
			values = [json.loads((root / f"curves/scores-{i * 3 + j}/{model}_-1.json").read_text())["macro_f1"] * 100 for j in range(3)]
			summary["learning_curves"][f"{week}/{model}"] = values
			axes[i].plot([25, 50, 100], values, "o-", label=label)
		axes[i].set(title=f"Evaluate 2022 week {week + 1}", xlabel="Training-game subset (%)", ylabel="Macro-F1 (%)", xticks=[25, 50, 100], ylim=(15, 60))
		axes[i].grid(alpha=0.2)
		axes[i].legend()
	fig.suptitle("Fixed 30-epoch learning curves · one seed · development only")
	fig.tight_layout()
	fig.savefig(args.output / "learning-curves.png", dpi=160)
	plt.close(fig)
	left = pd.concat([pd.read_parquet(root / f"objective/scores-{i}/neural-0_-1.parquet") for i in (0, 1)], ignore_index=True)
	right = pd.concat([pd.read_parquet(root / f"scores-{w}/neural-0_-1.parquet") for w in (3, 4)], ignore_index=True)
	summary["weighted_vs_reference_pooled_game_bootstrap"] = compare_predictions(left, right)
	# Class support and uncertainty are reported even when rare-class gains fail guards.
	for name, frame in (("reference", right), ("weighted", left)):
		metrics = score_predictions(frame.target.to_numpy(), frame[PROB_COLUMNS].to_numpy())
		predicted = frame[PROB_COLUMNS].to_numpy().argmax(1)
		games = list(frame.groupby(["season", "gameId"], sort=True).indices.values())
		matrices = np.stack([np.bincount(frame.target.to_numpy()[ids] * 7 + predicted[ids], minlength=49).reshape(7, 7) for ids in games])
		rng = np.random.default_rng(42)
		recalls = []
		for _ in range(2000):
			matrix = matrices[rng.integers(0, len(games), len(games))].sum(0)
			recalls.append(np.divide(np.diag(matrix), matrix.sum(1), out=np.full(7, np.nan), where=matrix.sum(1) > 0))
		for c, values in enumerate(metrics["per_class"].values()):
			values["predicted_count"] = int((predicted == c).sum())
			values["contributing_games"] = int(frame.loc[frame.target.eq(c), "gameId"].nunique())
			values["recall_ci95"] = np.nanquantile(np.asarray(recalls)[:, c], [0.025, 0.975]).tolist()
			values["campaign_support_resolved"] = values["contributing_games"] >= 10 and values["recall_ci95"][1] - values["recall_ci95"][0] <= 0.20
			values["bootstrap_replicates_without_class"] = int(np.isnan(np.asarray(recalls)[:, c]).sum())
		summary[f"pooled_{name}"] = metrics
	(args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
	for name, values in summary["arms"].items():
		print(name, {key: round(values[key] * 100, 2) for key in ("macro_precision", "macro_recall", "macro_f1", "binary_f1", "one_second_f1")})


if __name__ == "__main__":
	main()
