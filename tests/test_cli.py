import os
import subprocess
import sys
from pathlib import Path

import pytest


def invoke(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
	return subprocess.run([str(Path(sys.executable).parent / "nfl"), *args], cwd=cwd, capture_output=True, text=True, check=False)


def test_installed_help_works_outside_checkout(tmp_path: Path) -> None:
	result = invoke("--help", cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	assert "audit" in result.stdout
	assert list(tmp_path.iterdir()) == []


def test_missing_dataset_fails_without_creating_run(tmp_path: Path) -> None:
	result = invoke("audit", "--data-root", str(tmp_path / "missing"), "--run-dir", str(tmp_path / "run"), cwd=tmp_path)
	assert result.returncode != 0
	assert "data" in result.stderr.lower()
	assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("command", ["train", "predict", "evaluate", "benchmark", "experiment"])
def test_pending_commands_fail_clearly(command: str, tmp_path: Path) -> None:
	result = invoke(command, "--data-root", str(tmp_path), "--run-dir", str(tmp_path / "run"), cwd=tmp_path)
	assert result.returncode != 0
	assert "not implemented" in result.stderr.lower()
	assert not (tmp_path / "run").exists()


def test_import_has_no_output_side_effects(tmp_path: Path) -> None:
	env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", NFL_RUN_DIR=str(tmp_path / "run"))
	result = subprocess.run(
		[sys.executable, "-c", 'import cli; import common.paths; import models.transformer; assert "ray" not in __import__("sys").modules'],
		cwd=tmp_path,
		env=env,
		capture_output=True,
		text=True,
		check=False,
	)
	assert result.returncode == 0, result.stderr
	assert not (tmp_path / "run").exists()


def test_audit_reads_parquet_without_modifying_source(tmp_path: Path) -> None:
	import json

	import pyarrow as pa
	import pyarrow.parquet as pq

	raw = tmp_path / "raw"
	raw.mkdir()
	pq.write_table(pa.table({"gameId": [1, 2]}), raw / "games.parquet")
	before = (raw / "games.parquet").read_bytes()
	result = invoke("audit", "--data-root", str(raw), "--run-dir", str(tmp_path / "run"), cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	report = json.loads((tmp_path / "run" / "audit.json").read_text())
	assert report["files"][0]["rows"] == 2
	assert report["files"][0]["columns"] == ["gameId"]
	assert (raw / "games.parquet").read_bytes() == before
