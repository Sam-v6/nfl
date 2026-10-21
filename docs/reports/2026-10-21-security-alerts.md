# Dependency security fixes — October 21, 2026

Baseline: `894cc40` on `origin/main`. All 22 open Dependabot alerts matched vulnerable locked versions before the update; all 22 are outside their affected ranges afterward. No alerts were dismissed.

| Package | Before | After | GitHub alert numbers |
| --- | --- | --- | --- |
| anyio | 4.11.0 | 4.14.2 | #139, #140 |
| bleach | 6.3.0 | 6.4.0 | #77, #78, #79 |
| certifi | 2025.11.12 | 2026.10.19 | #167 |
| fsspec | 2025.10.0 | 2026.6.0 | #158 |
| msgpack | 1.1.2 | 1.2.1 | #80 |
| pillow | 12.3.0 | 13.0.0 | #168, #169, #170, #171, #172, #173, #174, #175, #176 |
| setuptools | 80.9.0 | 83.0.0 | #102 |
| torch | 2.10.0 | 2.13.0 | #85, #164 |
| uv | 0.12.17 | 0.13.4 | #165, #166 |

A broader PyPI advisory audit also found CVE-2026-113725 / GHSA-vxj4-q4g9-crj7 in virtualenv 21.14.5. The lock now uses 21.14.7 and its required packaging/distlib updates.

Torchvision moves from 0.25.0 to 0.28.0 to match Torch 2.13.0. XGBoost moves from 3.1.1 to 3.4.0 because older releases install CUDA 12 NCCL into the same files as Torch's CUDA 13 NCCL, causing `ImportError: undefined symbol: ncclCommResume` in an all-extras install. The updated lock has only the CUDA 13 NCCL package. The setuptools build floor is 83.0.0; uv pins in development, CI, README and required-version agree on the patched 0.13.4 line.

## Verification

- Fresh Python 3.12 environment: `uv sync --locked --all-extras` succeeded.
- `pytest -q`: **108 passed**, including the real Ray CPU trial and checkpoint restore.
- `ruff check src tests scripts`, `ruff format --check src tests scripts`, and `pyright`: passed (78 files formatted; zero type errors).
- `nfl --help`: passed.
- Torch/Torchvision imports, CPU torchvision NMS, and XGBoost CPU fit/predict: passed together after a fresh install.
- All 22 GitHub advisory version predicates: vulnerable before, clear after.
- PyPI advisory audit: **186 installed-platform dependency versions, zero known vulnerabilities**. Other-platform markers were skipped.
- Independent dependency review: no actionable findings.

To repeat the advisory audit without installing another tool in the project environment:

```bash
uv export --locked --all-extras --no-hashes --no-emit-project \
  --format requirements-txt --output-file /tmp/nfl-security-requirements.txt
uvx --from pip-audit pip-audit --no-deps --disable-pip \
  -r /tmp/nfl-security-requirements.txt
```

The exported requirements already enumerate the resolved transitive packages; `--no-deps --disable-pip` audits those exact versions. Audit results describe the advisory database at the time of the check.

## Limits

GitHub code scanning returned no analysis; secret scanning is disabled. The dependency changes do not establish coverage for either scanner. GPU execution is unverified: the local NVIDIA driver is unavailable. Torch now bundles CUDA 13.0 instead of 12.8, so GPU compatibility and previous GPU benchmark results require separate revalidation. Alerts remain open until the PR reaches the default branch and GitHub rescans.
