# Runtime and experiment decisions

The selected runtime is Python 3.12, Torch 2.9.1 with CUDA 12.8, Ray 2.59.0, uv 0.12.17, Ruff 0.14.7, and an exact `uv.lock`. Runtime, tuning, tracking, notebooks, legacy model, and development dependencies are separate. SciPy is explicit because temperature calibration imports it. MLflow remains optional; no second tracker or tuner was introduced.

Ray's function trials now call the same engine as local training, report directory checkpoints each epoch, and run through `Tuner.fit()`. A real two-epoch trial passed on the selected runtime. Distributed training is deferred: this is a small, single-GPU workload, and the unfinished migration did not provide verified distributed metrics. [Ray Trainable documentation](https://docs.ray.io/en/latest/tune/api/trainable.html).

Torch 2.13.0/CUDA 13.0 was tested in a separate environment, alongside Ray 2.59.0. CPU tests, a real Tune trial, and old-checkpoint CPU/GPU probability parity passed. It was rejected for the selected workload: matched FP32 eager training steps were 4.29 ms versus 2.51 ms on Torch 2.9.1; prepared-week batch prediction took 0.160 s versus 0.128 s, and batch-one prediction took 4.71 s versus 1.66 s. The small development score difference did not justify the regression. These are measured local workloads, not general claims about either version. [Official Torch builds](https://pytorch.org/get-started/previous-versions/).

FP32 training and inference are selected. With the stable validation path, BF16 training achieved development macro-F1 0.4674 versus FP32 0.4771, exceeding the allowed 0.005 regression. Temporal BF16 inference also failed its quality-parity gate. AMP, fused CUDA AdamW, TF32, compilation, and device-resident inputs are available for measured experiments, but none is enabled merely because the GPU supports it. TF32 is disabled in the selected comparison.

Compilation startup is reported separately. It improves steady training kernels, but startup can outweigh that saving in a short run. Batch inference is the main measured runtime gain. DataLoader and direct/device inputs now share exactly the same sampled order, preventing an apparent speed gain from being confounded by different training examples. CPU worker counts and transfer behavior are benchmarked rather than assumed beneficial.

Modern uv bootstrapping is explicit: `uvx --from uv==0.12.17 uv sync --locked --extra tuning`. The existing global uv installation was left untouched. [uv dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/).

## Autonomous execution controls

Each experiment holds a per-run lock and, for CUDA, a shared GPU lock inherited by its worker. The controller runs one GPU allocation at a time, persists attempts and GPU/wall time, preserves counters on resume, validates returned bundles, and records `complete`, `failed`, `blocked`, or `budget_exhausted`. Explicit CUDA cannot silently fall back to CPU. Ray search can change only model hyperparameters; device/backend/epochs/sample ceilings remain controller-owned. Trials have a 60-minute limit and a ten-second checkpoint shutdown grace, charged to the ledger; a hard kill follows if necessary.

The final configuration imports the verified development-training receipt count and elapsed times. Older benchmarks did not record complete process wall time, so a conservative one-hour allowance covers their startup, measurement, and diagnostic overhead. The receipt inventory and hashes are saved with local run artifacts. This is an explicit conservative accounting estimate, not a claim that older benchmark time was measured exactly. Overall ceilings remain 24 trials and 24 GPU-hours, with half reserved for confirmation.

Final selection freezes source/configuration/data/benchmark hashes before reading final features. Recovery verifies that identity and reuses hashed, atomically committed stages. It cannot alter the candidate or repeat a completed scoring stage. Temperature and abstention thresholds use Week 6 only. Promotion verifies successful gates and the exact evaluated calibration metadata, then publishes a content-addressed read-only bundle and atomic pointer. A negative final result is terminal research evidence, not a reason to retune on the final weeks.

Manual CI integration prepares raw data, trains two epochs, resumes to three, loads portable weights, calibrates, predicts, and evaluates Week 5. It performs no final-role inference or scoring; preparation uses a bounded Week 1/5/6 subset. CUDA is opt-in so CI does not automatically compete with rendering. A separate real Ray trial exercises checkpoint reporting. Fast PR CI requires neither tracking data nor a GPU.

## Measured delivery outcome

The final selected model remains spatial FP32, TF32 off and compilation off. Matched prepared-week inference improved 12.9× on GPU and 4.2× on CPU through batching; the later shared-host GPU run varied substantially. Compiled FP32 training steps improved from 2.705 to 2.331 ms, but incurred a 7.23 s cold step, and full-training 1.5× acceleration was not established. These are workload measurements, not claims about all models or hardware. The [frozen report](2026-10-11-coverage-comparison/README.md) contains precision/recall, all selection failures and timing boundaries.

CI pins uv 0.12.17 and Python 3.12, with an isolated tool-bin directory to avoid the self-hosted runner selecting its old global uv. CPU verification does not request the GPU. Its GitHub display name remains `test-model-training-and-prediction-generation` to satisfy the existing required-check rule. See the [delivery index](2026-10-11-delivery.md) for exact-head checks and merge status.

Post-evaluation review repairs ensure resume honors a checkpoint already at the early-stop limit, Ray chooses the best reported epoch matching exported weights, and worker bundle export can recover after interruption. The frozen six local confirmation runs were unaffected. Historical source identity remains in the selection receipt; a modified source tree must start a separately identified experiment.
