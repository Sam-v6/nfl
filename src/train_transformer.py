"""Ray Tune adapter for the same engine used by the installed ``nfl train`` CLI.

Single-GPU function trials avoid an unverified distributed metric path. DDP can
be added after a workload demonstrates a benefit and global metrics are tested.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from ray.tune import ResultGrid


def tune_trainable(config: dict[str, Any], manifest_path: str) -> None:
	from ray import tune

	from training import train_model

	run_dir = Path(tune.get_context().get_trial_dir()) / "training"
	checkpoint = tune.get_checkpoint()
	settings = dict(config)
	if checkpoint is not None:
		with checkpoint.as_directory() as restored:
			shutil.copytree(restored, run_dir, dirs_exist_ok=True)
		settings["resume"] = True

	def report(metrics: dict[str, Any], directory: Path) -> None:
		tune.report(metrics, checkpoint=tune.Checkpoint.from_directory(str(directory)))

	train_model(settings, Path(manifest_path), run_dir, reporter=report)


def run_tune(config: dict[str, Any], manifest_path: Path, output_dir: Path, num_samples: int = 1) -> ResultGrid:
	from ray import tune
	from ray.tune.schedulers import ASHAScheduler

	if not 1 <= num_samples <= 24:
		raise ValueError("Tune trial count must be between 1 and 24")
	output_dir = output_dir.resolve()
	if output_dir.exists():
		raise ValueError("Tune output directory already exists; restore the experiment explicitly")
	trainable = tune.with_parameters(tune_trainable, manifest_path=str(manifest_path.resolve()))
	resources: dict[str, float] = {"cpu": 2.0, "gpu": 1.0 if config.get("device") == "cuda" else 0.0}
	tuner = tune.Tuner(
		tune.with_resources(trainable, resources),
		param_space=config,
		tune_config=tune.TuneConfig(
			metric="macro_f1",
			mode="max",
			num_samples=num_samples,
			max_concurrent_trials=1,
			scheduler=ASHAScheduler(max_t=int(config.get("epochs", 30)), grace_period=min(5, int(config.get("epochs", 30))), reduction_factor=3, stop_last_trials=False),
		),
		run_config=tune.RunConfig(
			storage_path=str(output_dir.parent),
			name=output_dir.name,
			verbose=0,
			checkpoint_config=tune.CheckpointConfig(num_to_keep=2, checkpoint_score_attribute="macro_f1", checkpoint_score_order="max"),
		),
	)
	return tuner.fit()
