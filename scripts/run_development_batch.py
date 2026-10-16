"""Execute a predeclared batch, carrying forward the campaign compute receipt."""

import argparse
import json
from pathlib import Path

from dataset import file_hash
from experiment import RunLock, run_experiment


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("plan", type=Path, help="JSON list of manifest/trials jobs")
	parser.add_argument("prior_receipt", type=Path)
	parser.add_argument("output", type=Path)
	args = parser.parse_args()
	prior = json.loads(args.prior_receipt.read_text())
	if prior["status"] != "complete":
		raise ValueError("Prior stage must finish before starting a new stage")
	jobs = json.loads(args.plan.read_text())
	if not 1 <= len(jobs) <= 12:
		raise ValueError("Require 1–12 predeclared jobs")
	root = args.output.resolve()
	chain_path = root.parent / "campaign-chain.json"
	with RunLock(root.parent / "campaign-chain.lock"):
		if chain_path.exists():
			chain = json.loads(chain_path.read_text())
			if chain["active"] or chain["receipt_sha256"] != file_hash(args.prior_receipt):
				raise ValueError("Stale budget receipt or unfinished batch; inspect the campaign chain before recovery")
		chain_path.write_text(json.dumps({"active": str(root), "receipt_sha256": file_hash(args.prior_receipt)}))
		root.mkdir(parents=True, exist_ok=False)
		(root / "plan.json").write_text(args.plan.read_text())
		(root / "prior.json").write_text(json.dumps({"path": str(args.prior_receipt.resolve()), "sha256": file_hash(args.prior_receipt), "receipt": prior}, indent=2) + "\n")
		for number, job in enumerate(jobs):
			config = {
				"manifest": job["manifest"],
				"trials": job["trials"],
				"max_trials": 36,
				"gpu_hours": 4,
				"reserve_hours": 0,
				"trial_seconds": 1800,
				"wall_seconds": 14400,
				"initial_attempts": prior["attempts"],
				"initial_gpu_seconds": prior["gpu_seconds"],
			}
			path = root / f"job-{number:02d}.json"
			path.write_text(json.dumps(config, indent=2) + "\n")
			result = run_experiment(path, root / f"job-{number:02d}")
			prior = json.loads(result.read_text())
			(root / "terminal.json").write_text(json.dumps(prior, indent=2) + "\n")
			print(number, prior, flush=True)
			if prior["status"] != "complete":
				break
		chain_path.write_text(json.dumps({"active": None, "receipt_sha256": file_hash(root / "terminal.json")}))


if __name__ == "__main__":
	main()
