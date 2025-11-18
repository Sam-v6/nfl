FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1

# System deps (Python 3.12 is default on Ubuntu 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    build-essential pkg-config git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install uv #TODO: pin version?
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

WORKDIR /app

# Pre-copy lock/manifest for cached dep layer (works with uv or requirements)
COPY pyproject.toml uv.lock* requirements.txt* ./

# (Optional) Pre-install torch CUDA wheels here to cache, or keep in pyproject/requirements.
# Uncomment if you want to pin at build time:
# RUN uv pip install --index-url https://download.pytorch.org/whl/cu124 \
#     torch torchvision torchaudio

# Sync/install project deps
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of your repo
COPY . .

# Set a sensible default; override in docker-compose.yml
CMD ["bash"]