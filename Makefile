PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
PROFILE ?= smoke
SEED ?= 0
BACKEND ?= hf

.PHONY: install tasks test test-all lint sft-data sft prefs dpo rl-sparse rl-dense eval report all docker-build clean

install:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

tasks:
	$(PY) -m gwa.env.tasks --dump

test:
	$(PY) -m pytest -q -m "not slow"

test-all:
	$(PY) -m pytest -q

coverage:
	$(PY) -m pytest -q -m "not slow" --cov=gwa --cov-report=term-missing

lint:
	.venv/bin/ruff check .

sft-data:
	$(PY) -m gwa.cli build-sft --profile $(PROFILE) --seed $(SEED)

sft:
	$(PY) -m gwa.cli train-sft --profile $(PROFILE) --seed $(SEED)

prefs:
	$(PY) -m gwa.cli build-prefs --profile $(PROFILE) --seed $(SEED)

dpo:
	$(PY) -m gwa.cli train-dpo --profile $(PROFILE) --seed $(SEED)

rl-sparse:
	$(PY) -m gwa.cli train-rl --mode sparse --profile $(PROFILE) --seed $(SEED) --backend $(BACKEND)

rl-dense:
	$(PY) -m gwa.cli train-rl --mode dense --profile $(PROFILE) --seed $(SEED) --backend $(BACKEND)

eval:
	$(PY) -m gwa.cli eval --profile $(PROFILE) --seed $(SEED) --backend $(BACKEND)

report:
	$(PY) -m gwa.cli report --profile $(PROFILE)

all: tasks sft-data sft prefs dpo rl-sparse rl-dense eval report

docker-build:
	docker build -t gridworldagentrl .

clean:
	rm -rf checkpoints data results/*.jsonl results/*.json results/*.md results/figures/*.png .pytest_cache
