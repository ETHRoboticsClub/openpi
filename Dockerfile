# Dockerfile for PyTorch training (PI0/PI05 models with multi-GPU DDP support).
# Based on the existing serve_policy.Dockerfile pattern.
#
# Build:
#   docker build -t openpi_train -f Dockerfile .
#
# Run (single GPU):
#   docker run --rm -it --gpus=all -v .:/app openpi_train \
#     python scripts/train_pytorch.py debug --exp_name my_experiment
#
# Run (multi-GPU DDP):
#   docker run --rm -it --gpus=all --ipc=host -v .:/app openpi_train \
#     torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#     scripts/train_pytorch.py pi0_aloha_sim --exp_name my_experiment

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Install system dependencies (git-lfs needed by LeRobot, build tools for native extensions).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git git-lfs linux-headers-generic build-essential clang && \
    rm -rf /var/lib/apt/lists/*

# Copy from the cache instead of linking since it's a mounted volume.
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install dependencies using the lockfile (without installing the project itself).
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Patch transformers with custom model implementations.
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r /tmp/transformers_replace/* {} \
    && rm -rf /tmp/transformers_replace

# Shared memory size is critical for multi-GPU DDP training (use --ipc=host or --shm-size).
CMD ["/bin/bash"]
