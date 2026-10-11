# NFL Coverage Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate reliable seven-class pre-snap coverage prediction, accelerate training/inference, and make execution reproducible and unattended.

**Architecture:** Keep the existing spatial transformer as a baseline. Share causal feature preparation between training and inference, add versioned datasets and model bundles, and compare a small temporal model only after evaluation and performance foundations pass.

**Tech Stack:** Python 3.12, uv, PyTorch, pandas/Polars/Parquet, scikit-learn/LightGBM, Ray Tune/ASHA, MLflow, pytest, Ruff, Pyright.

**Spec:** `docs/superpowers/specs/2026-10-10-coverage-modernization.md`.

**Status:** APPROVED FOR IMPLEMENTATION on October 11, 2026. Progress is recorded in the execution ledger.

## Global Constraints

- Scope confirmed in chat: **pre-snap prediction first; post-snap identification later.**
- Version the class order as `Cover-0, Cover-1, Cover-2, Cover-3, Quarters, 2-Man, Cover-6`, retaining current numeric IDs. Preserve original labels beside canonical labels.
- Pre-snap models receive only measurements available at or before the prediction cutoff, strictly before the snap. Labels may describe eventual coverage; they are targets, never input features.
- Split by whole games before augmentation; augmentation is training-only.
- Keep Python 3.12 initially; use exact compatible lockfile versions.
- Use branches `Sam-v6/<descriptive-name>` and worktrees under `.worktrees/<descriptive-name>`.
- Preserve root edits; use read-only raw data and isolated generated outputs.
- Proposed budget: one GPU job at a time; 24 total GPU-hours; at most 24 HPO trials total; at most 100 epochs and 60 minutes per search trial; three seeds for finalists; three repair attempts for the same failing gate. Reserve half the compute budget for confirmation and reporting.
- Approval to implement is not automatically approval to merge; default delivery is reviewable PRs unless merge authority is explicitly included.

## Review Focus

- Future measurements or outcome columns accidentally enter earlier predictions: Task 2 adds a future-perturbation test and strict feature allowlist.
- Short 2021 history masquerades as observed motion: Tasks 2 and 6 test sequence masks and forbid backward fill.
- Rare/unknown classes disappear from evaluation denominators: Tasks 2 and 3 test fixed class vocabulary, exclusion receipts, and per-class support.
- An interrupted Ray run resumes the wrong weights, repeats data, or reports local-rank metrics as global: Tasks 4 and 8 exercise checkpoint/resume and fault receipts.
- Batching or concurrency misaligns frame IDs or overwrites artifacts: Tasks 5 and 8 compare keyed predictions, batch tails, duplicate detection, and immutable run directories.

## Working sequence

Tasks 1–4 establish trustworthy, runnable baselines. Task 5 addresses the strongest observed performance opportunity. Task 6 is the principal modeling experiment. Tasks 7–8 make bounded execution and delivery reliable. Implement in this order; do not start large HPO before Tasks 1–5 pass. Conditional research extensions in the spec are not part of the first batch.

Each task should produce one bounded PR or a small group of independently reviewable PRs if its diff is too large. Use the existing `.github/pull_request_template.md`. Reviewers should check test meaning, causal validity, artifact provenance, and acceptance evidence. Do not run two agents against the same mutable output directory or GPU.

## Task 1: Preserve work and establish fast verification

**Files:** Modify `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/pr-ci.yml`, `src/common/args.py`, `src/common/paths.py`; create `src/cli.py`, `tests/conftest.py`, `tests/test_cli.py`, `tests/test_paths.py`; update `uv.lock` only after resolving dependency compatibility.

**Interfaces:** Real console command `nfl` dispatches `audit`, `prepare`, `train`, `predict`, `evaluate`, `benchmark`, `experiment`. Initially only `audit` and help need to work; unimplemented commands must fail clearly. Shared command inputs: `--data-root`, `--run-dir`, `--config`. `--device` accepts `cpu`, `cuda`, or `auto` where computation is needed. Modules must be importable without reading datasets, starting Ray, or creating output directories. Use the existing source layout; avoid a package-wide move solely for style.

- [ ] Inspect fresh Git state and save a provenance receipt of the current commit and dirty-file hashes. Create the implementation worktree from the feature branch's current tip, after checking remote divergence. Copy its existing training-script changes into the worktree as an explicitly identified carry-forward; keep the root file and notebook untouched. Do not quietly start from older main and lose feature-branch work.
- [ ] Add tests: CLI help exits zero from outside the checkout; explicit paths resolve correctly; importing CLI/models creates no files; requesting a missing dataset gives a useful nonzero failure. Run `uv run pytest tests/test_cli.py tests/test_paths.py -q` and confirm the intended failures.
- [ ] Add pytest and Pyright development dependencies, correct the placeholder entry point, and implement the path/command surface. Keep heavy runtime/tuning imports lazy. Fix Ruff findings only in maintained code involved in this work; scope initial typing to the maintained pipeline and new modules.
- [ ] Split CI into fast CPU verification and an explicit bounded real-data/GPU integration job. The CPU job must need neither LFS data nor CUDA and must run lint, format, type checks, and pytest. The integration job must use an isolated output directory and explicit resource timeout.
- [ ] Verify `uv sync --locked`, `uv run nfl --help`, focused tests, Ruff, formatting, and Pyright. Record the GPU-access diagnostic without attempting driver installation. Commit and review.

## Task 2: Audit labels and build a causal dataset contract

**Files:** Create `src/coverage_labels.py`, `src/dataset.py`, `src/audit_data.py`, `configs/coverage.json`, `tests/test_coverage_labels.py`, `tests/test_dataset.py`; modify `src/load_data.py`, `src/clean_data.py`, `src/create_features.py`, `src/cli.py`.

**Interfaces:** `canonicalize_coverage(raw: str | None) -> int | None` uses the fixed seven-class order. `audit_data(data_root: Path, output_dir: Path) -> Path` emits a JSON report and play-level Parquet inventory. `prepare_dataset(data_root: Path, config: dict, output_dir: Path) -> Path` emits `manifest.json`, split/example indexes, and float32 tensor shards. Every example index carries season/game/play/cutoff IDs, target, split, observed-history length, and augmentation parent. Manifest includes input hashes, class/feature schema, split membership, exclusions, and preprocessing version.

- [ ] Add exact mapping tests: `Cover-3 Seam -> 3`, `Cover 6-Left -> 6`, `2-Man -> 5`, `Quarters -> 4`, and `Bracket/None -> None`. Assert no mutation of caller data and preserve source labels.
- [ ] Add invariants: no game overlap between splits; augmentation children inherit training split; modifying any frame later than a cutoff leaves its input tensor unchanged; post-snap columns are rejected as model inputs; reflection negates `v_y` and double reflection restores inputs; duplicate player/frame keys fail; a bad frame cannot silently null an entire week. Run these tests to see the intended failures.
- [ ] Implement the spec's split allocation and endpoint policy. Keep evaluation rows original/unaugmented. Build a pure feature-preparation function usable without coverage or outcome columns; apply retrospective cohort/label selection outside that function.
- [ ] For snapshots, preserve the original five features as a versioned baseline schema. For temporal inputs, use up to 30 observed frames ending at cutoff, left padding plus a time mask; no backward fill. Stable player keys must not be derived from future positions. Initially exclude frames with incomplete 22-player membership and record the reason.
- [ ] Add fixtures with half-second history, missing snap, unknown labels, duplicate joins, gaps, and incomplete players. Require input hashes/schema mismatch to invalidate caches. Seed augmentation explicitly and ensure two identical preparations produce identical indexes and tensors.
- [ ] Run `uv run nfl audit --data-root <raw> --run-dir <audit>` and inspect counts against the planning audit, explaining differences due to quality filters. Run `uv run pytest tests/test_coverage_labels.py tests/test_dataset.py -q`. Freeze the manifest and split IDs before model experiments. Commit and review.

## Task 3: Define play-level evaluation and cheap comparators

**Files:** Create `src/evaluate.py`, `src/models/baselines.py`, `tests/test_evaluate.py`, `tests/test_baselines.py`; modify `src/process_predictions.py`, `src/cli.py`.

**Interfaces:** `evaluate_predictions(predictions: pd.DataFrame, manifest_path: Path, output_dir: Path) -> Path` emits `metrics.json`, class-support and confusion tables, calibration plots, and exclusion counts. Prediction rows are uniquely keyed by `(season, gameId, playId, cutoff_frame)` with seven ordered probability columns. `compare_predictions(candidate: pd.DataFrame, baseline: pd.DataFrame, seed: int = 42) -> dict` uses 2,000 paired game-bootstrap replicates and a fixed class list. Metadata records excluded replicates if a statistic is undefined.

- [ ] Add fixtures with hand-checkable perfect, wrong, imbalanced, and missing-class predictions. Assert macro-F1 includes all seven classes, probability validation rejects NaN/bad sums, support is counted by play, duplicate prediction rows fail, and paired comparisons require matching example keys.
- [ ] Add tests that a model whose rare-class recall collapses cannot pass by increasing majority-class accuracy, and that uncertainty/abstention reports retain the full eligible denominator. Fewer than 20 final examples in a class must mark its claim insufficient.
- [ ] Implement fixed-horizon metrics from the spec. Fit logistic and LightGBM comparators using only training features; geometric summaries must be permutation-invariant and available by cutoff. Default priors come from training only. Use the same cohort and feature availability across comparisons.
- [ ] Implement forward-chaining development checks inside Weeks 1–5. Keep final evaluation command separate from tuning and require a frozen-selection receipt before it runs. Final results cannot be read by the experiment-selection routine.
- [ ] Run `uv run pytest tests/test_evaluate.py tests/test_baselines.py -q`; run the CPU baseline on the frozen development manifest; verify report denominators and loss values manually on a tiny fixture. Commit and review.

## Task 4: Repair training and deliver seven-class snapshots

**Files:** Modify `src/train_transformer.py`, `src/models/transformer.py`, `src/common/mlflow.py`; create `src/model_bundle.py`, `tests/test_training.py`, `tests/test_model_bundle.py`, `tests/test_tune_smoke.py`.

**Interfaces:** `train_model(config: dict, manifest_path: Path, run_dir: Path) -> Path` returns a bundle directory. `load_bundle(path: Path, device: torch.device) -> tuple[nn.Module, dict]` verifies schema/vocabulary and loads portable eager weights. `predict_proba(model: nn.Module, features: torch.Tensor) -> torch.Tensor` returns `[batch, classes]`. Binary legacy artifacts require explicit schema metadata rather than guessing by filename. Bundle separates best-evaluation weights from latest-resume checkpoint.

- [ ] Write CPU tests: a tiny separable dataset overfits; seven-class logits have correct shape; save/load preserves predictions; class-order mismatch fails; player permutation leaves eval predictions unchanged within tolerance; a final worse epoch cannot replace the best model. Run to confirm failures.
- [ ] Consolidate duplicate training paths into a shared engine. Make input/output dimensions schema-driven. Add CPU FP32, hardware-checked BF16, and FP16 with GradScaler; snapshot scaler/optimizer/RNG/sampler state. Ensure loss/accuracy totals are sample-weighted. Preserve legacy binary loading and train a corrected binary comparator from scratch under the new split.
- [ ] Make Tune actually execute and report each trial, using the chosen Ray version's supported API. Use directory checkpoints and a real best-checkpoint policy. Compile after loading portable weights; catch runtime compilation failures explicitly and record an eager fallback, rather than hiding them.
- [ ] Add a one-trial/two-epoch CPU Tune test with synthetic data. Require completed training iterations, finite metrics, restored checkpoint parity, and isolated trial outputs. Add interrupt/resume comparison with an uninterrupted deterministic CPU run. If DDP is retained, add a two-worker CPU test whose global metrics match a single-process reference, including uneven validation size.
- [ ] Run focused tests, then a bounded real-data development run. Compare seven-class spatial, binary, and cheap baselines without accessing final scores. Keep a completed baseline even if the neural model loses. Commit and review.

## Task 5: Batch inference and establish performance evidence

**Files:** Modify `src/generate_predictions.py`, `src/process_predictions.py`; create `src/benchmark.py`, `tests/test_prediction_batches.py`, `tests/test_benchmark.py`; wire commands in `src/cli.py`.

**Interfaces:** `predict_dataset(bundle_path: Path, manifest_path: Path, output_dir: Path, batch_size: int) -> Path` writes uniquely keyed prediction Parquet. It must also accept an unlabeled prepared dataset. `benchmark(config: dict, output_dir: Path) -> Path` records environment, hardware, data hashes, warmup, repeats, batch size, precision, compile mode, end-to-end wall time, throughput, latency percentiles, and memory.

- [ ] Add tests comparing frame-by-frame eager reference output with batches of 1, 7, and 64, including a nondivisible tail and shuffled input keys. Assert no dropped/duplicated frames, numerical parity, and correct probability-column order. Add a second-run test proving results do not append stale rows.
- [ ] Replace repeated week-table filtering with one grouped tensorization per bounded chunk. Use actual batched `torch.inference_mode()` inference, bounded transfers, and keyed output writes. Keep downstream tracking merges separate from the compact prediction table.
- [ ] Add a synthetic benchmark-timer test and real same-model benchmarks. Separate cold load/compile, warm batch-one latency, bulk model throughput, and full weekly wall time. Use 5 warmups and 30 timed repeats for model microbenchmarks; 3 repeated runs for end-to-end measurements. Synchronize CUDA around timing boundaries, and record data-cache state.
- [ ] Compare direct tensor indexing to DataLoader collation; compare workers 0/2/4 on the target host; remove per-batch `.item()` synchronization where measured. Benchmark eager versus compile and FP32 versus supported mixed precision independently. Do not change model architecture in a performance-only comparison.
- [ ] Run `uv run pytest tests/test_prediction_batches.py tests/test_benchmark.py -q`. Require parity before timing claims. Record actual results versus the 5× inference/1.5× training targets; if CUDA remains unavailable, mark GPU evidence blocked and retain CPU evidence. Commit and review.

## Task 6: Test motion history and incremental football features

**Files:** Create `src/models/temporal.py`, `src/football_features.py`, `tests/test_temporal.py`, `tests/test_football_features.py`; modify dataset/model factory/training interfaces without changing bundle contracts.

**Interfaces:** `TemporalCoverageModel.forward(x: Tensor, time_mask: Tensor) -> Tensor` consumes `[batch, time, 22, features]` and a `[batch, time]` boolean mask, producing seven logits. Reuse frame-level spatial encoding then a unidirectional GRU; padding must be excluded from normalization/pooling and recurrent state updates. `build_context_features(tracking_prefix: pd.DataFrame, context: dict) -> np.ndarray` uses only cutoff-available quantities and a persisted feature order.

- [ ] Write tests for padding invariance, no future-frame dependence, permutation invariance within frames, short-history acceptance without invented observations, and finite gradients. A frame after cutoff, future player order, or missing-history pad must not change a prediction. Run tests to confirm intended failures.
- [ ] Implement the compact temporal model and allowlisted context features. Train on 2022 initially. Choose all preprocessing statistics on fit data. Use line-of-scrimmage metadata only when available before the prediction time; do not use `playClockAtSnap`, eventual assignments, target receiver, or post-play outcomes.
- [ ] Run ordered development ablations: snapshot vs temporal on matching data; temporal plus geometric/context features; optional 2021 snapshot initialization versus training from scratch. Compare ordinary CE with capped class weighting. Keep a run ledger so each change can be attributed.
- [ ] Select at most two finalists using development metrics and the common-cohort rule. If temporal/context additions do not improve development results, keep the spatial/tabular baseline. Do not automatically move on to a larger transformer or player-assignment task.
- [ ] Run `uv run pytest tests/test_temporal.py tests/test_football_features.py -q`; record sample counts, macro-F1/log loss, compute, and latency for each ablation. Commit and review.

## Task 7: Compatible upgrades and bounded experiment control

**Files:** Modify `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `.github/workflows/pr-ci.yml`; create `src/experiment.py`, `configs/experiment.json`, `tests/test_experiment.py`; update training/benchmark adapters only as required.

**Interfaces:** `run_experiment(config_path: Path, output_dir: Path) -> Path` manages a persisted `ledger.json` and terminal receipt. Budget counters are cumulative across resume. Run states are `complete`, `failed`, `budget_exhausted`, `blocked`. Use a per-run lock and reserve GPU ownership before dispatch. Every trial names its input manifest, code revision, runtime config, and parent experiment.

- [ ] Add tests with a fake clock/trainer: trial cap, wall/GPU-hour cap, recovery after interruption, stale lock handling with process-liveness checks, output collision, failed trial, missing GPU, and failed artifact validation. A restarted process must not reset its budget.
- [ ] Implement bounded ASHA search with the spec's limits. Configure graceful checkpointing and hard subprocess timeout; reserve budget for three-seed confirmation. Do not silently fall back to expensive CPU training when CUDA was explicitly requested.
- [ ] Audit imports and classify dependencies into runtime/dev/optional groups. Upgrade one compatible cohort at a time: uv/tooling, Torch stack, Ray/MLflow, dataframe/ML libraries. Consult official version-specific docs and record the selected versions; do not blindly select development releases or add a second tuner/tracker. Remove a dependency only when import and workflow checks demonstrate it is unused.
- [ ] After each cohort, run CPU verification, checkpoint compatibility, Tune smoke test, and same-host benchmark comparison. Reject upgrades that break verified artifacts or materially degrade the selected workload; document a pinned dependency as a valid outcome.
- [ ] Verify `uv run pytest tests/test_experiment.py -q`, the full fast suite, and one bounded end-to-end experiment with forced interruption/resume. Commit and review.

## Task 8: Calibrate, run locked evaluation, and deliver

**Files:** Create `src/calibrate.py`, `tests/test_calibrate.py`, `tests/test_promotion.py`; modify `src/evaluate.py`, `src/process_predictions.py`, `README.md`; add generated, versioned reports under `artifacts/runs/<run-id>/` and a concise tracked report under `docs/reports/`.

**Interfaces:** `fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float` fits one positive scalar on Week 6 only. `check_promotion(candidate_metrics: dict, baseline_metrics: dict, comparison: dict) -> dict` returns the gate decisions and evidence references. Promotion creates an atomic pointer to an immutable accepted bundle only when every required gate passes; otherwise it leaves the incumbent intact. Test-only fixture promotion must never write a real model pointer.

- [ ] Write tests: calibration preserves argmax; invalid temperatures fail; a synthetic overconfident case improves log loss; final split IDs are rejected by calibration; missing benchmark/seed/class evidence prevents promotion; a worse candidate cannot replace the incumbent. Run to confirm the intended failures.
- [ ] Freeze selected architectures, hyperparameters, runtime settings, split hashes, and seeds 17/42/73. Train seed-matched finalists within remaining budget. Calibrate each on Week 6. Fit any confidence threshold there and report its accuracy/coverage tradeoff without claiming novel-class detection.
- [ ] Run the locked Weeks 7–8 evaluation once for frozen candidates and baselines. Apply the spec's confidence-interval, class recall, binary, calibration, and secondary-endpoint gates. Failure means a documented negative/inconclusive result, not another tuning round. Historical Week 9 results remain separately labeled.
- [ ] Produce seven-class confusion/calibration figures, accuracy and macro-F1 by pre-snap horizon, counts/exclusions, uncertainty examples, and before/after performance tables. Extend existing play visualizations with top coverage probabilities. Do not describe attention weights as causal explanations or incorrect predictions as proven defensive disguise.
- [ ] Run full fast CI and bounded real-data integration from a clean worktree-local environment. Verify a model bundle can load and predict without training data or labels. Inject one interruption to confirm the experiment ledger and final receipt distinguish incomplete from successful work.
- [ ] Deliver PRs, artifact locations, exact reproduction commands, remaining limitations, and pass/fail gate results. The project must remain usable with the baseline if no candidate passes. Do not claim GPU acceleration, improved predictive power, or remote CI success without the corresponding evidence. Commit and obtain the planned independent review before delivery.

## Plan self-review

- Scope: multiclass capability, predictive experiments, runtime performance, dependency/tooling refresh, and autonomous execution all have owning tasks.
- Data: known coverage labels exist, but class imbalance, short 2021 history, historical holdout exposure, and pass-cohort selection remain explicit.
- Interfaces: all prediction/evaluation paths use the same vocabulary, keyed records, manifest, and model bundle; temporal input changes do not change output semantics.
- Budget: work can terminate with a valid baseline and an honest inconclusive report; no success criterion requires an open-ended research search.
- Review status: implementation has not started. Approval is the next step.
