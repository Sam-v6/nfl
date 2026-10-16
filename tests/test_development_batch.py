import importlib
import json
import sys
from pathlib import Path

import pytest


def test_batch_rejects_reused_budget_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
	module = importlib.import_module("run_development_batch")
	prior = tmp_path / "prior.json"
	prior.write_text(json.dumps({"status": "complete", "attempts": 12, "gpu_seconds": 300}))
	plan = tmp_path / "plan.json"
	plan.write_text(json.dumps([{"manifest": "fixture", "trials": [{"device": "cuda"}]}]))

	def fake_run(config: Path, output: Path) -> Path:
		settings = json.loads(config.read_text())
		output.mkdir()
		result = output / "terminal.json"
		result.write_text(json.dumps({"status": "complete", "attempts": settings["initial_attempts"] + 1, "gpu_seconds": settings["initial_gpu_seconds"] + 10}))
		return result

	monkeypatch.setattr(module, "run_experiment", fake_run)
	monkeypatch.setattr(sys, "argv", ["run", str(plan), str(prior), str(tmp_path / "first")])
	module.main()
	monkeypatch.setattr(sys, "argv", ["run", str(plan), str(prior), str(tmp_path / "second")])
	with pytest.raises(ValueError, match="receipt"):
		module.main()
	assert not (tmp_path / "second").exists()
