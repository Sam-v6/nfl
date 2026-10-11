from pathlib import Path

import pytest


def test_explicit_paths_are_resolved_without_creating_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	from common.paths import RunPaths

	monkeypatch.chdir(tmp_path)
	paths = RunPaths.resolve(Path("raw"), Path("run"))
	assert paths.data_root == tmp_path / "raw"
	assert paths.run_dir == tmp_path / "run"
	assert not paths.run_dir.exists()
	assert not paths.data_root.exists()
