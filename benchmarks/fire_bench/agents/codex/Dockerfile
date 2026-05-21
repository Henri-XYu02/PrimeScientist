# Per-run runtime image for the fire_bench codex agent.
# Mirrors the openhands Dockerfile (CUDA 12.1 + Python ML stack) and adds
# Node.js + the codex CLI plus the Python packages fire_bench tasks commonly
# import. Built once with:
#   docker build -t firebench-codex:0.1 benchmarks/fire_bench/agents/codex/
#
# At run time, agents/codex/run.py and run_inherit.py dispatch through
# `docker run --rm -v <sandbox>:/sandbox firebench-codex:0.1 codex ...`,
# so any pip install / file write the agent does goes into the container's
# ephemeral writable layer and disappears on `--rm`. Only `/sandbox` (mounted
# from the host) survives the run.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Base utilities + Python + Node.js
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        git curl gnupg ca-certificates build-essential \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# Python deps — same baseline as openhands (numpy/pandas/datasets/transformers)
# plus everything the fire_bench tasks regularly reach for.
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
        numpy pandas datasets transformers tokenizers \
        openai anthropic google-generativeai \
        python-docx python-dotenv toml \
        scipy scikit-learn matplotlib \
        tqdm huggingface-hub

# PyTorch + torchvision with CUDA 12.1 wheels
RUN python3 -m pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Pre-install codex CLI globally so we don't pay an npx download per run.
RUN npm install -g @openai/codex@0.39.0


# Writable HOME for the agent. World-writable + sticky so it works regardless
# of the --user UID we map at run time. Anything the agent puts here (pip
# --user installs, codex config, scratch files) lives in the container's
# ephemeral layer and disappears on `docker run --rm`.
RUN mkdir -p /agent-home && chmod 1777 /agent-home

# The host sandbox is mounted here at run time.
WORKDIR /sandbox
