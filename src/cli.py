"""Explicit, isolated entry points for coverage experiments."""

import argparse
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
		sub.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
	args = parser.parse_args(argv)
	paths = RunPaths.resolve(args.data_root, args.run_dir)
	try:
		if args.command == "audit":
			from audit_data import audit_data

			print(audit_data(paths.data_root, paths.run_dir))
		else:
			parser.error(f"{args.command} is not implemented yet")
	except (OSError, ValueError) as exc:
		parser.error(str(exc))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
