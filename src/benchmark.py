"""Same-model measurements; cold startup, steady kernels and input pipeline differ."""

import copy
import json
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import file_hash
from dataset_io import load_examples
from evaluate import score_predictions
from generate_predictions import predict_batches, predict_dataset
from model_bundle import load_bundle
from models.factory import forward_model
from training import select_device


def measure(operation: Callable[[], None], synchronize: Callable[[], None], warmups: int = 5, repeats: int = 30, clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
	if warmups < 0 or repeats < 1:
		raise ValueError("Invalid benchmark repetition count")
	for _ in range(warmups):
		operation()
	synchronize()
	timings = []
	for _ in range(repeats):
		synchronize()
		started = clock()
		operation()
		synchronize()
		timings.append(clock() - started)
	return {"warmups": warmups, "repeats": repeats, "median_ms": float(np.median(timings) * 1000), "p95_ms": float(np.percentile(timings, 95) * 1000), "total_seconds": sum(timings)}


def benchmark(config: dict[str, Any], output_dir: Path) -> Path:
	if output_dir.exists():
		raise ValueError("Benchmark output already exists")
	total_started = time.perf_counter()
	torch.set_num_threads(int(config.get("cpu_threads", 4)))
	torch.set_float32_matmul_precision("highest")
	device = select_device(str(config.get("device", "auto")))
	bundle = Path(config["bundle"])
	manifest = Path(config["manifest"])
	batch_size = int(config.get("batch_size", 512))
	warmups, repeats = int(config.get("warmups", 5)), int(config.get("repeats", 30))
	started = time.perf_counter()
	eager, metadata = load_bundle(bundle, device)
	if device.type == "cuda":
		torch.cuda.synchronize()
	cold_load = time.perf_counter() - started
	x, mask, y, index = load_examples(manifest, "tune", offset=-1)
	x = x.contiguous()
	reference = predict_batches(eager, x, mask, batch_size, device)
	baseline_quality = score_predictions(y.numpy(), reference.numpy())
	output_dir.mkdir(parents=True)
	os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output_dir / "compiler-cache"))

	def synchronize() -> None:
		if device.type == "cuda":
			torch.cuda.synchronize()

	modes = []
	precisions = ["fp32", "bf16"] if device.type == "cuda" and torch.cuda.is_bf16_supported() else ["fp32"]
	for compiled in config.get("compile_modes", [False, True] if device.type == "cuda" else [False]):
		for precision in precisions:
			model = copy.deepcopy(eager)
			cold_start = time.perf_counter()
			try:
				if compiled:
					torch.compiler.reset()
					model = cast(nn.Module, torch.compile(model))
				probs = predict_batches(model, x, mask, batch_size, device, precision)
				synchronize()
				cold_seconds = time.perf_counter() - cold_start
				quality = score_predictions(y.numpy(), probs.numpy())
				if precision == "fp32":
					parity = bool(torch.allclose(probs, reference, atol=1e-5, rtol=1e-4))
				else:
					parity = quality["macro_f1"] >= baseline_quality["macro_f1"] - 0.005 and quality["log_loss"] <= baseline_quality["log_loss"] * 1.01
				if not parity:
					modes.append({"compile": bool(compiled), "precision": precision, "status": "parity_failed", "quality": quality})
					continue
				latencies = {}
				for size in (1, min(batch_size, len(x))):
					inputs = x[:size].to(device)
					inputs_mask = mask[:size].to(device)

					dtype = torch.bfloat16 if precision == "bf16" else torch.float32

					def operation(model: nn.Module = model, inputs: torch.Tensor = inputs, inputs_mask: torch.Tensor = inputs_mask, dtype: torch.dtype = dtype, precision: str = precision) -> None:
						with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
							forward_model(model, inputs, inputs_mask)

					measured = measure(operation, synchronize, warmups, repeats)
					measured["examples_per_second"] = size * 1000 / measured["median_ms"]
					latencies[str(size)] = measured

				modes.append(
					{
						"compile": bool(compiled),
						"precision": precision,
						"status": "complete",
						"parity": True,
						"cold_compile_and_first_prediction_seconds": cold_seconds,
						"latencies": latencies,
						"quality": quality,
					}
				)
			except (RuntimeError, ValueError) as exc:
				modes.append({"compile": bool(compiled), "precision": precision, "status": "failed", "reason": str(exc)})
	# DataLoader workers are compared against direct slices on the identical resident tensors.
	pipelines = []
	for workers in (None, 0, 2, 4):
		loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=bool(workers)) if workers is not None else None

		def transfer(loader: DataLoader | None = loader) -> None:
			batches = loader if loader is not None else ((x[i : i + batch_size], y[i : i + batch_size]) for i in range(0, len(x), batch_size))
			for features, targets in batches:
				features.to(device, non_blocking=True)
				targets.to(device, non_blocking=True)

		pipelines.append({"workers": workers, **measure(transfer, synchronize, 1, 3)})
	# Steady training on fixed real inputs, without validation or checkpoint I/O.
	training_modes = []
	for compiled in config.get("compile_modes", [False, True] if device.type == "cuda" else [False]):
		for precision in precisions:
			model = copy.deepcopy(eager).train()
			if compiled:
				torch.compiler.reset()
				model = cast(nn.Module, torch.compile(model))
			optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, fused=device.type == "cuda")
			inputs, targets = x[:batch_size].to(device), y[:batch_size].to(device)
			inputs_mask = mask[:batch_size].to(device)
			dtype = torch.bfloat16 if precision == "bf16" else torch.float32

			def step(
				model: nn.Module = model,
				optimizer: torch.optim.Optimizer = optimizer,
				inputs: torch.Tensor = inputs,
				targets: torch.Tensor = targets,
				inputs_mask: torch.Tensor = inputs_mask,
				dtype: torch.dtype = dtype,
				precision: str = precision,
			) -> None:
				optimizer.zero_grad(set_to_none=True)
				with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
					loss = nn.functional.cross_entropy(forward_model(model, inputs, inputs_mask), targets)
				loss.backward()
				optimizer.step()

			cold_step = time.perf_counter()
			step()
			synchronize()
			training_modes.append({"compile": bool(compiled), "precision": precision, "cold_step_seconds": time.perf_counter() - cold_step, **measure(step, synchronize, warmups, repeats)})
	weekly = []
	for size in (1, batch_size):
		timings = []
		row_counts = []
		for repeat in range(3):
			start = time.perf_counter()
			path = predict_dataset(bundle, manifest, output_dir / f"weekly-{size}-{repeat}", size, split="tune", device=str(device))
			timings.append(time.perf_counter() - start)
			row_counts.append(json.loads((path.parent / "terminal.json").read_text())["rows"])
		weekly.append(
			{
				"batch_size": size,
				"rows": row_counts[-1],
				"offsets": "all configured pre-snap horizons",
				"repeats": 3,
				"seconds": timings,
				"median_seconds": float(np.median(timings)),
			}
		)
	report = {
		"status": "complete",
		"hardware": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
		"device": str(device),
		"torch": torch.__version__,
		"cuda_runtime": torch.version.cuda,
		"python": platform.python_version(),
		"cpu_threads": torch.get_num_threads(),
		"tf32": bool(torch.backends.cuda.matmul.allow_tf32),
		"elapsed_seconds": time.perf_counter() - total_started,
		"batch_size": batch_size,
		"cold_load_seconds": cold_load,
		"cache_state": "uncontrolled OS cache; warm tensor cache after verified loading",
		"manifest_sha256": file_hash(manifest),
		"weights_sha256": metadata["weights_sha256"],
		"model_config": metadata["config"],
		"model_source_sha256": metadata.get("source_sha256", {}),
		"inference_modes": modes,
		"transfer_pipelines": pipelines,
		"training_modes": training_modes,
		"weekly_end_to_end": weekly,
		"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
	}
	path = output_dir / "benchmark.json"
	path.write_text(json.dumps(report, indent=2) + "\n")
	return path
