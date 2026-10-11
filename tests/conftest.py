"""Keep CPU fixtures small and deterministic without reserving a GPU."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
