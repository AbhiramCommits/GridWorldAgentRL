"""Figures generated only from measured result files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .utils import REPO_ROOT, ensure_parent  # noqa: E402


def _read_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def plot_learning_curves(
    results_dir: str | Path = "results",
    out_path: str | Path = "results/figures/rl_curves.png",
) -> Path:
    results = Path(results_dir)
    if not results.is_absolute():
        results = REPO_ROOT / results
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"sparse": "#d95f02", "dense": "#1b9e77"}
    any_data = False
    for mode in ("sparse", "dense"):
        rows = _read_curve(results / f"rl_{mode}_curve.jsonl")
        if not rows:
            continue
        any_data = True
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r["mean_reward"] for r in rows], marker="o", label=mode,
                     color=colors[mode])
        axes[1].plot(steps, [r["mean_kl"] for r in rows], marker="o", label=mode,
                     color=colors[mode])
    axes[0].set_title("GRPO mean episode reward")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("mean reward")
    axes[1].set_title("KL vs frozen reference")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("mean KL")
    for ax in axes:
        ax.grid(alpha=0.25)
        if any_data:
            ax.legend()
    fig.tight_layout()
    out = ensure_parent(out_path)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
