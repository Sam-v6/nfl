import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def configuration(tmp_path: Path, **updates: object) -> Path:
	path = tmp_path / "experiment.json"
	path.write_text(
		json.dumps(
			{
				"manifest": str(tmp_path / "manifest.json"),
				"gpu_lock": str(tmp_path / "gpu.lock"),
				"trials": [{"device": "cpu", "epochs": 2}],
				"max_trials": 2,
				"gpu_hours": 24,
				"reserve_hours": 12,
				**updates,
			}
		)
	)
	return path


def test_resume_does_not_reset_trial_or_compute_counters(tmp_path: Path) -> None:
	from experiment import run_experiment

	now = [0.0]
	calls = []

	def trainer(settings: dict, manifest: Path, directory: Path, timeout: float) -> Path:
		calls.append(settings)
		now[0] += 4
		raise RuntimeError("injected failure")

	config = configuration(tmp_path)
	output = tmp_path / "run"
	receipt = run_experiment(config, output, trainer=trainer, clock=lambda: now[0], validate=lambda path: None)
	assert json.loads(receipt.read_text())["status"] == "failed"
	data = json.loads(config.read_text())
	data["resume"] = True
	config.write_text(json.dumps(data))
	receipt = run_experiment(config, output, trainer=trainer, clock=lambda: now[0], validate=lambda path: None)
	ledger = json.loads((output / "ledger.json").read_text())
	assert ledger["attempts"] == 2
	assert ledger["elapsed_seconds"] == 8
	assert json.loads(receipt.read_text())["status"] == "failed"
	run_experiment(config, output, trainer=trainer, clock=lambda: now[0], validate=lambda path: None)
	assert len(calls) == 2
	assert json.loads(receipt.read_text())["status"] == "budget_exhausted"


def test_missing_explicit_gpu_blocks_without_training(tmp_path: Path) -> None:
	from experiment import run_experiment

	config = configuration(tmp_path, trials=[{"device": "cuda", "epochs": 2}])
	result = run_experiment(config, tmp_path / "run", gpu_available=lambda: False)
	assert json.loads(result.read_text())["status"] == "blocked"
	assert json.loads((result.parent / "ledger.json").read_text())["attempts"] == 0


def test_reservation_and_timeout_are_enforced(tmp_path: Path) -> None:
	from experiment import run_experiment

	timeouts = []

	def trainer(settings: dict, manifest: Path, directory: Path, timeout: float) -> Path:
		timeouts.append(timeout)
		return directory

	config = configuration(tmp_path, gpu_hours=0.01, reserve_hours=0.009, trials=[{"device": "cuda", "epochs": 2}])
	result = run_experiment(config, tmp_path / "run", trainer=trainer, gpu_available=lambda: True, validate=lambda path: None)
	assert json.loads(result.read_text())["status"] == "complete"
	assert timeouts[0] == pytest.approx(3.6)


def test_collision_and_invalid_artifact_fail_closed(tmp_path: Path) -> None:
	from experiment import run_experiment

	config = configuration(tmp_path)
	output = tmp_path / "collision"
	output.mkdir()
	with pytest.raises(ValueError, match="exists"):
		run_experiment(config, output)

	def trainer(settings: dict, manifest: Path, directory: Path, timeout: float) -> Path:
		return directory / "missing"

	result = run_experiment(config, tmp_path / "bad", trainer=trainer)
	assert json.loads(result.read_text())["status"] == "failed"


def test_live_lock_refuses_concurrent_owner_and_stale_metadata_recovers(tmp_path: Path) -> None:
	from experiment import RunLock

	path = tmp_path / "run.lock"
	path.write_text(json.dumps({"pid": 999999999}))
	with RunLock(path):
		with pytest.raises(RuntimeError, match="owned"):
			with RunLock(path):
				pass
	with RunLock(path):
		assert json.loads(path.read_text())["pid"] > 0


def test_controller_resumes_real_cpu_checkpoint_after_interrupt(tmp_path: Path) -> None:
	from test_model_bundle import MODEL_CONFIG
	from test_training import training_fixture

	from experiment import run_experiment
	from training import train_model

	manifest = training_fixture(tmp_path / "data")
	config = configuration(tmp_path, manifest=str(manifest), trials=[{**MODEL_CONFIG, "device": "cpu", "epochs": 4}])
	attempts = [0]

	def trainer(settings: dict, manifest: Path, directory: Path, timeout: float) -> Path:
		attempts[0] += 1

		def interruption(metrics: dict, path: Path) -> None:
			if metrics["epoch"] == 1:
				raise KeyboardInterrupt("injected interrupted coordinator")

		return train_model(settings, manifest, directory, reporter=interruption if attempts[0] == 1 else None)

	output = tmp_path / "run"
	path = run_experiment(config, output, trainer=trainer)
	assert json.loads(path.read_text())["status"] == "failed"
	data = json.loads(config.read_text())
	data["resume"] = True
	config.write_text(json.dumps(data))
	path = run_experiment(config, output, trainer=trainer)
	assert json.loads(path.read_text())["status"] == "complete"
	assert json.loads(path.read_text())["attempts"] == 2
	assert json.loads((output / "trial-000/terminal.json").read_text())["epochs_completed"] == 4


def test_wall_cap_persists_and_unknown_device_is_rejected(tmp_path: Path) -> None:
	from experiment import run_experiment

	now = [0.0]

	def trainer(settings: dict, manifest: Path, directory: Path, timeout: float) -> Path:
		now[0] += 5.0
		return directory

	config = configuration(tmp_path, wall_seconds=5.0, trials=[{"device": "cpu"}, {"device": "cpu"}])
	path = run_experiment(config, tmp_path / "run", trainer=trainer, clock=lambda: now[0], validate=lambda path: None)
	assert json.loads(path.read_text())["status"] == "budget_exhausted"
	bad = configuration(tmp_path, trials=[{"device": "typo"}])
	with pytest.raises(ValueError, match="device"):
		run_experiment(bad, tmp_path / "bad")


def test_search_space_cannot_override_resource_or_control_fields(tmp_path: Path) -> None:
	from experiment import run_experiment

	for field, choices in [("device", ["cuda"]), ("epochs", [101]), ("num_samples", [25]), ("backend", ["ray"])]:
		config = configuration(tmp_path, trials=[{"device": "cpu", "backend": "ray", "search_space": {field: choices}}])
		with pytest.raises(ValueError, match="search"):
			run_experiment(config, tmp_path / field)


def test_ray_worker_exports_best_epoch_and_recovers_existing_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	import experiment
	import train_transformer

	trials = []
	for name, scores in [("A", [0.9, 0.1]), ("B", [0.5, 0.4])]:
		checkpoint_dir = tmp_path / name
		(checkpoint_dir / "bundle").mkdir(parents=True)
		(checkpoint_dir / "bundle/weights.pt").write_text(name)
		checkpoint = SimpleNamespace(as_directory=lambda path=checkpoint_dir: nullcontext(str(path)))
		trials.append(SimpleNamespace(error=None, checkpoint=checkpoint, scores=scores))

	def best_result(metric: str, mode: str, scope: str = "last") -> SimpleNamespace:
		return max(trials, key=lambda trial: max(trial.scores) if scope == "all" else trial.scores[-1])

	monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(tune=SimpleNamespace()))
	monkeypatch.setattr(train_transformer, "run_tune", lambda *args: SimpleNamespace(get_best_result=best_result))
	config = configuration(tmp_path)
	output = tmp_path / "output"
	output.mkdir()
	monkeypatch.setattr(sys, "argv", ["experiment", "--worker", str(config), str(tmp_path / "manifest.json"), str(output)])
	experiment._worker()
	assert (output / "bundle/weights.pt").read_text() == "A"
	(output / "bundle/weights.pt").write_text("interrupted export")
	(output / "bundle/stale.bin").write_text("prior bundle")
	experiment._worker()
	assert (output / "bundle/weights.pt").read_text() == "A"
	assert not (output / "bundle/stale.bin").exists()


def test_approved_campaign_stops_at_thirty_six_attempts(tmp_path: Path) -> None:
	from experiment import run_experiment

	calls = []

	def trainer(settings: dict, manifest: Path, directory: Path, timeout: float) -> Path:
		calls.append(directory)
		return directory

	config = configuration(tmp_path, max_trials=36, initial_attempts=35, trials=[{"device": "cpu"}, {"device": "cpu"}])
	result = run_experiment(config, tmp_path / "campaign", trainer=trainer, validate=lambda _: None)
	assert len(calls) == 1
	assert json.loads(result.read_text())["status"] == "budget_exhausted"
	assert json.loads((result.parent / "ledger.json").read_text())["attempts"] == 36
