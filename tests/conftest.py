"""Keep CPU fixtures small and deterministic without reserving a GPU."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nfl-matplotlib")
# Ray workers reuse the test interpreter instead of rebuilding a `uv run` environment.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
