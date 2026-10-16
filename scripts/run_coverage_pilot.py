"""Run the predeclared two-fold, six-fit pilot under a shared 30-minute GPU cap."""

import argparse
import json
from pathlib import Path

from development import build_fold
from experiment import run_experiment


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("source", type=Path)
	parser.add_argument("output", type=Path)
	args = parser.parse_args()
	root = args.output.resolve()
	root.mkdir(parents=True, exist_ok=False)
	attempts, seconds = 0, 0.0
	for week in (3, 4):
		manifest = build_fold(args.source.resolve(), root / f"fold-{week}", week)
		trials = [
			{"epochs": epochs, "sampling": sampling, "precision": "fp32", "device": "cuda", "input_pipeline": "loader", "cpu_threads": 4, "seed": 42, "patience": 10}
			for sampling, epochs in (("endpoint", 30), ("endpoint", 100), ("play", 100))
		]
		config = {
			"manifest": str(manifest),
			"max_trials": 6,
			"gpu_hours": 0.5,
			"reserve_hours": 0,
			"trial_seconds": 600,
			"wall_seconds": 1800,
			"initial_attempts": attempts,
			"initial_gpu_seconds": seconds,
			"trials": trials,
		}
		path = root / f"pilot-{week}.json"
		path.write_text(json.dumps(config, indent=2) + "\n")
		result = run_experiment(path, root / f"pilot-{week}")
		ledger = json.loads((result.parent / "ledger.json").read_text())
		attempts, seconds = ledger["attempts"], ledger["gpu_seconds"]
		print(result.read_text(), flush=True)
		if json.loads(result.read_text())["status"] != "complete":
			break


if __name__ == "__main__":
	main()
