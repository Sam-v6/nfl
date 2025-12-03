#!/usr/bin/env python
"""
Defines common filesystem paths used across the project.
"""

from pathlib import Path


def project_root() -> Path:
	"""
	Finds the repository root directory.

	Inputs:
	- None.

	Outputs:
	- root_path: Path pointing to the project root.
	"""

	return Path(__file__).resolve().parents[2]


PROJECT_ROOT = project_root()

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INFERENCE_DIR = PROJECT_ROOT / "data" / "inference"
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
