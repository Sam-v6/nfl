"""Persisted compute ceilings and one owner for isolated, resumable experiments."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Self, TextIO

from model_bundle import load_bundle

# Same as train_transformer: keep Ray workers on this interpreter under `uv run`.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


class RunLock:
	"""Kernel locks release on process death
	stale PID metadata is informational."""

	def __init__(self, path: Path) -> None:
		self.path = path
		self.stream: TextIO | None = None

	def __enter__(self) -> Self:
		self.path.parent.mkdir(parents=True, exist_ok=True)
		stream = self.path.open("a+")
		try:
			fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
		except BlockingIOError as exc:
			stream.close()
			raise RuntimeError("Resource already owned by a live process") from exc
		stream.seek(0)
		stream.truncate()
		stream.write(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}))
		stream.flush()
		self.stream = stream
		return self

	def __exit__(self, *args: object) -> None:
		if self.stream is not None:
			self.stream.close()


def _write(path: Path, value: dict[str, Any]) -> None:
	temporary = path.with_suffix(".tmp")
	temporary.write_text(json.dumps(value, indent=2) + "\n")
	temporary.replace(path)


def _validate(path: Path) -> None:
	import torch

	load_bundle(path, torch.device("cpu"))


def _gpu_available() -> bool:
	import torch

	return torch.cuda.is_available()


def _subprocess_train(settings: dict[str, Any], manifest: Path, directory: Path, timeout: float) -> Path:
	settings = dict(settings)
	lock_fds = tuple(settings.pop("_lock_fds", ()))
	config_path = directory.parent / (directory.name + "-config.json")
	config_path.write_text(json.dumps(settings, indent=2) + "\n")
	command = [sys.executable, "-m", "cli", "train", "--data-root", str(manifest.parent), "--run-dir", str(directory), "--config", str(config_path)]
	if settings.get("backend") == "ray":
		command = [sys.executable, "-m", "experiment", "--worker", str(config_path), str(manifest), str(directory)]
	with (directory.parent / (directory.name + ".log")).open("a") as log:
		process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=lock_fds)
		try:
			code = process.wait(timeout=timeout)
		except BaseException:
			os.killpg(process.pid, signal.SIGINT)
			try:
				process.wait(timeout=10)
			except subprocess.TimeoutExpired:
				os.killpg(process.pid, signal.SIGKILL)
				process.wait()
			raise
	if code != 0:
		raise RuntimeError(f"Training subprocess failed with exit {code}; see {directory.name}.log")
	return directory / "bundle"


def run_experiment(
	config_path: Path,
	output_dir: Path,
	*,
	trainer: Callable[[dict[str, Any], Path, Path, float], Path] = _subprocess_train,
	clock: Callable[[], float] = time.time,
	gpu_available: Callable[[], bool] = _gpu_available,
	validate: Callable[[Path], None] = _validate,
) -> Path:
	config = json.loads(config_path.read_text())
	resume = bool(config.pop("resume", False))
	identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
	max_trials = int(config.get("max_trials", 24))
	gpu_hours, reserve_hours = float(config.get("gpu_hours", 24)), float(config.get("reserve_hours", 12))
	trial_seconds = float(config.get("trial_seconds", 3600))
	wall_seconds = float(config.get("wall_seconds", 86400))
	initial_gpu_seconds = float(config.get("initial_gpu_seconds", 0))
	initial_attempts = int(config.get("initial_attempts", 0))
	if not (
		1 <= max_trials <= 36
		and math.isfinite(gpu_hours)
		and 0 < gpu_hours <= 24
		and 0 <= reserve_hours < gpu_hours
		and 0 < trial_seconds <= 3600
		and 0 < wall_seconds <= 86400
		and 0 <= initial_gpu_seconds <= gpu_hours * 3600
		and 0 <= initial_attempts <= max_trials
	):
		raise ValueError("Invalid experiment ceilings")
	trials = config["trials"]
	if not trials or any(not 1 <= int(trial.get("epochs", 30)) <= 100 for trial in trials):
		raise ValueError("Require trials bounded to 1–100 epochs")
	if any(trial.get("device", "cuda") not in ("auto", "cpu", "cuda") or not 1 <= int(trial.get("num_samples", 1)) <= 24 for trial in trials):
		raise ValueError("Invalid trial device or sample count")
	searchable = {"model_dim", "num_heads", "num_layers", "multiplier", "dropout", "lr", "weight_decay", "batch_size"}
	for trial in trials:
		search = trial.get("search_space", {})
		if set(search) - searchable or any(not isinstance(choices, list) or not choices for choices in search.values()):
			raise ValueError("Invalid search space: only model hyperparameters may be searched")
	if output_dir.exists() and not resume:
		raise ValueError("Experiment output already exists; resume explicitly")
	output_dir.mkdir(parents=True, exist_ok=resume)
	manifest = Path(config["manifest"]).resolve()
	terminal = output_dir / "terminal.json"
	with RunLock(output_dir / "run.lock") as run_lock:
		ledger_path = output_dir / "ledger.json"
		if resume:
			ledger = json.loads(ledger_path.read_text())
			if ledger["config_hash"] != identity:
				raise ValueError("Resume experiment configuration mismatch")
		else:
			ledger = {"config_hash": identity, "attempts": initial_attempts, "gpu_seconds": initial_gpu_seconds, "elapsed_seconds": 0.0, "completed": [], "active": None, "trials": []}
		if ledger["active"] is not None:
			# A dead coordinator cannot prove when its allocation stopped. Charge
			# the reservation conservatively rather than resetting its budget.
			active = ledger["active"]
			charged = min(active["timeout"], max(0.0, clock() - active["started"]))
			ledger["elapsed_seconds"] += charged
			if active["gpu"]:
				ledger["gpu_seconds"] += charged
			ledger["active"] = None
			_write(ledger_path, ledger)
		status, reason = "complete", None
		for number, trial in enumerate(trials):
			if number in ledger["completed"]:
				continue
			settings = dict(trial)
			device = settings.get("device", "cuda")
			uses_gpu = device == "cuda" or (device == "auto" and gpu_available())
			settings["device"] = "cuda" if uses_gpu else "cpu"
			if device == "cuda" and not gpu_available():
				status, reason = "blocked", "Explicit CUDA is unavailable; no CPU fallback started"
				break
			cost = int(settings.get("num_samples", 1)) if settings.get("backend") == "ray" else 1
			available = gpu_hours * 3600 - ledger["gpu_seconds"]
			if config.get("stage", "search") == "search":
				available -= reserve_hours * 3600
			if ledger["attempts"] + cost > max_trials or ledger["elapsed_seconds"] >= wall_seconds or (uses_gpu and available <= 0):
				status, reason = "budget_exhausted", "Cumulative trial or compute ceiling reached"
				break
			timeout = min(trial_seconds, wall_seconds - ledger["elapsed_seconds"], available if uses_gpu else trial_seconds)
			directory = output_dir / f"trial-{number:03d}"
			settings["resume"] = (directory / "latest.pt").exists() or any(directory.glob("experiment_state*.json"))
			gpu_lock = RunLock(Path(config.get("gpu_lock", f"/tmp/nfl-gpu-{os.getuid()}.lock"))) if uses_gpu else nullcontext()
			try:
				with gpu_lock:
					started = clock()
					ledger["attempts"] += cost
					ledger["active"] = {"number": number, "started": started, "timeout": timeout, "gpu": uses_gpu}
					_write(ledger_path, ledger)
					try:
						invocation = dict(settings)
						if trainer is _subprocess_train:
							assert run_lock.stream is not None
							invocation["_lock_fds"] = [run_lock.stream.fileno()]
							if isinstance(gpu_lock, RunLock):
								assert gpu_lock.stream is not None
								invocation["_lock_fds"].append(gpu_lock.stream.fileno())
						bundle = trainer(invocation, manifest, directory, timeout)
						validate(bundle)
						ledger["completed"].append(number)
						ledger["trials"].append({"number": number, "status": "complete", "bundle": str(bundle), "config": settings})
					finally:
						elapsed = max(0.0, clock() - started)
						ledger["elapsed_seconds"] += elapsed
						ledger["gpu_seconds"] += elapsed if uses_gpu else 0
						ledger["active"] = None
						_write(ledger_path, ledger)
			except BaseException as exc:
				status = "blocked" if isinstance(exc, RuntimeError) and "owned" in str(exc) else "failed"
				reason = str(exc) or type(exc).__name__
				ledger["trials"].append({"number": number, "status": status, "reason": reason, "config": settings})
				break
		_write(ledger_path, ledger)
		_write(
			terminal,
			{
				"status": status,
				"reason": reason,
				"completed_trials": len(ledger["completed"]),
				"attempts": ledger["attempts"],
				"gpu_seconds": ledger["gpu_seconds"],
				"elapsed_seconds": ledger["elapsed_seconds"],
				"config_hash": identity,
			},
		)
	return terminal


def _worker() -> None:
	from ray import tune

	from train_transformer import run_tune

	settings = json.loads(Path(sys.argv[2]).read_text())
	search = settings.pop("search_space", {})
	for name, choices in search.items():
		settings[name] = tune.choice(choices)
	directory = Path(sys.argv[4])
	results = run_tune(settings, Path(sys.argv[3]), directory, int(settings.pop("num_samples", 1)))
	best = results.get_best_result(metric="macro_f1", mode="max", scope="all")
	if best.error is not None or best.checkpoint is None:
		raise RuntimeError("Ray search produced no valid checkpoint")
	import shutil

	with best.checkpoint.as_directory() as checkpoint:
		destination = directory / "bundle"
		if destination.exists():
			shutil.rmtree(destination)
		shutil.copytree(Path(checkpoint) / "bundle", destination)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
	_worker()
