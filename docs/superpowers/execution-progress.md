# Approved coverage modernization execution

Goal activated October 11, 2026. Worktree: `.worktrees/coverage-modernization`; branch: `Sam-v6/coverage-modernization`; base: `fc898aa`.

- Task 1 complete at `a03d4d0`: isolated CLI/path handling, read-only input inventory, pytest/Pyright, installed packaging, fast CPU CI, manual data integration. 11 tests, Ruff/format, and Pyright passed.
- Task 2 implemented: fixed vocabulary, game-separated split contract, causal masked prefixes, train-only augmentation and corrected velocity reflection. Real-data dataset preparation completed with verified index/source hashes, 16,457 original primary-endpoint plays, and 1,851 final comparison plays. Dataset hash: `07e30dd6dbee417298224f05db66a90be36f54d7c4fb33de29edd37c5f234ab0`. Missing-cutoff exclusions count play/horizon opportunities, while unknown-label and missing-snap exclusions count plays.
- Task 3 implemented: verified play-level reports, fixed seven-class metrics, game bootstrap, final-data gate, verified dataset reader, geometry baselines, and forward folds. Week 5 development scores at -0.1s: logistic accuracy 52.62%, macro-F1 0.4082; LightGBM accuracy 52.34%, macro-F1 0.4369. These are development results, not final or comparable to the historical binary headline.
- Task 4 implemented: one shared local/Ray trainer, eager portable bundles with explicit binary compatibility, separate best/latest checkpoints, exact interrupted CPU resume, and a real two-epoch Ray trial. The 30-epoch BF16 development run completed on CUDA: seven-class macro-F1 0.4833, versus LightGBM 0.4369. Corrected binary macro-F1 is 0.7796. Both are Week 5 development results; final superiority is not established.
- Tasks 5–8 pending: inference/performance evidence; temporal ablations; bounded search/upgrades; calibration/final delivery.

Host GPU verified: RTX 4090 with 24 GB memory, idle at the check. Sandbox visibility differs from host access; use approved host execution for GPU checks and jobs. Host PyTorch confirms CUDA 12.8 and BF16 support. Two bounded GPU training runs completed, using 20.97 seconds of trainer time in total. Check rendering/process activity before substantial jobs.

The uncommitted root training changes were copied into this worktree with provenance; only duplicate imports, annotation, and formatting were repaired in Task 1. Runtime repair belongs to Task 4. The root notebook and training script remain unchanged; original SHA-256 values are in `provenance.json`.

Rulings: user-specified worktree placement takes precedence over native placement; initial integration audits data until new training guarantees output isolation; worktree-local environment prevents imports from the primary checkout. Proposed 24 GPU-hour / 24-trial ceilings apply.

Detailed step evidence and rulings live in `.superpowers/sdd/2026-10-10-coverage-modernization/progress.md` until final delivery.
