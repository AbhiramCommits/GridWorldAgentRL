"""Shared utilities: determinism, configuration, git metadata, and JSONL IO."""

from __future__ import annotations

import json
import os
import random
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA) for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic kernels where available. CPU generation is deterministic already.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    profile: str | None = None,
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a YAML config, resolve ``extends``, and apply a named profile.

    Profiles are deep-merged over the top-level configuration. Unknown profiles
    are rejected so a typo never silently trains an unintended size.
    """
    cfg_path = Path(path) if path is not None else CONFIG_DIR / "default.yaml"
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if "extends" in raw:
        parent_path = (cfg_path.parent / raw.pop("extends")).resolve()
        with open(parent_path, encoding="utf-8") as fh:
            parent = yaml.safe_load(fh) or {}
        raw = _deep_merge(parent, raw)

    profiles = raw.get("profiles", {})
    selected = profile or raw.get("default_profile", "default")
    if selected not in profiles:
        raise ValueError(f"Unknown profile {selected!r}; available: {sorted(profiles)}")
    cfg = _deep_merge(raw, profiles[selected])
    cfg["profile"] = selected
    cfg.pop("profiles", None)
    cfg["config_path"] = str(cfg_path)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def git_commit_hash(short: bool = True) -> str:
    """Return the current git commit hash, or ``"unknown"`` outside a repo."""
    try:
        cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        return subprocess.check_output(cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_metadata(cfg: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Standard provenance block attached to every result artifact."""
    meta = {
        "profile": cfg.get("profile"),
        "seed": cfg.get("seed"),
        "model_id": cfg.get("model", {}).get("id"),
        "git_commit": git_commit_hash(),
        "config_path": cfg.get("config_path"),
    }
    meta.update(extra)
    return meta


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class JsonlWriter:
    """Tiny append-only JSONL writer that creates parent directories."""

    def __init__(self, path: str | Path):
        self.path = ensure_parent(path)
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=True, default=_json_default) + "\n")

    def write_many(self, records: Iterable[dict[str, Any]]) -> int:
        n = 0
        for rec in records:
            self.write(rec)
            n += 1
        return n

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: str | Path, payload: Any) -> Path:
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0
