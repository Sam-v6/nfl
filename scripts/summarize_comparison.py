"""Export a completed frozen comparison as portable tables and scientific figures."""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--run-dir", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	args = parser.parse_args()
	report = json.loads((args.run_dir / "terminal.json").read_text())
	if report["status"] != "complete":
		parser.error("Require a completed frozen comparison")
	args.output_dir.mkdir(parents=True, exist_ok=True)
	os.environ.setdefault("MPLCONFIGDIR", "/tmp/nfl-matplotlib")
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	rows = report["seed_results"]
	keys = sorted(rows[0]["candidate"], key=lambda k: int(k.split("/")[1]))
	offsets = np.array([int(k.split("/")[1]) / 10 for k in keys])
	fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
	for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Seven-class macro-F1"), strict=True):
		for name, label, color in [("baseline", "LightGBM", "#b76e00"), ("candidate", "Spatial transformer", "#225ea8")]:
			values = np.array([[row[name][key][metric] for key in keys] for row in rows])
			ax.plot(offsets, values.mean(0), "o-", label=label, color=color)
			ax.fill_between(offsets, values.min(0), values.max(0), color=color, alpha=0.15)
		ax.set(xlabel="Seconds before snap", ylabel=title, title=title, xticks=offsets, ylim=(0, 0.7))
		ax.grid(alpha=0.2)
		ax.legend()
	counts = [rows[0]["candidate"][key]["eligible_plays"] for key in keys]
	fig.suptitle("Frozen Weeks 7–8 comparison: mean and range over seeds 17 / 42 / 73")
	fig.text(0.5, 0.01, f"Endpoint cohorts differ: n = {counts}. Range is seed variation, not a confidence interval.", ha="center", fontsize=9)
	fig.tight_layout(rect=(0, 0.04, 1, 0.93))
	fig.savefig(args.output_dir / "horizons.png", dpi=160)
	plt.close(fig)
	chosen = next(row for row in rows if row["seed"] == 42)
	classes = list(chosen["candidate"]["final/-1"]["per_class"])
	fig, ax = plt.subplots(figsize=(10, 4.5))
	positions = np.arange(len(classes))
	for name, shift, label, color in [("baseline", -0.2, "LightGBM", "#b76e00"), ("candidate", 0.2, "Spatial transformer", "#225ea8")]:
		ax.bar(positions + shift, [chosen[name]["final/-1"]["per_class"][key]["recall"] for key in classes], width=0.4, label=label, color=color)
	ax.set(xticks=positions, xticklabels=classes, ylabel="Recall", ylim=(0, 1), title="Preselected seed 42, 0.1 seconds before snap")
	ax.legend()
	fig.tight_layout()
	fig.savefig(args.output_dir / "class-recall.png", dpi=160)
	plt.close(fig)
	for name in ("candidate", "baseline"):
		for kind in ("confusion", "calibration"):
			shutil.copyfile(args.run_dir / f"seed-42/{name}-final-report/{kind}_final_-1.png", args.output_dir / f"{name}-{kind}.png")
	(args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
	shutil.copyfile(args.run_dir / "selection.json", args.output_dir / "selection.json")


if __name__ == "__main__":
	main()
