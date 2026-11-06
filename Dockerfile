FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GWA_NUM_THREADS=4

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, then the project (torch is already satisfied).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY configs ./configs
COPY Makefile ./
RUN mkdir -p results/figures data checkpoints assets

ENTRYPOINT ["python", "-m", "gwa.cli"]
