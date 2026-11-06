"""Bootstrap confidence intervals and paired comparisons."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``mean``, ``ci_low``, ``ci_high``, ``n``, and ``n_resamples``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "n_resamples": n}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(arr.size),
        "n_resamples": int(n),
    }


def paired_bootstrap_diff(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    n: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap for ``mean(a - b)``.

    ``p_a_better`` is the fraction of resamples whose mean difference is > 0.
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size != bb.size:
        raise ValueError("paired bootstrap requires equal-length inputs")
    if aa.size == 0:
        return {
            "mean_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_a_better": 0.0,
            "n": 0,
            "n_resamples": n,
        }
    diffs = aa - bb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n, diffs.size))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_a_better": float((means > 0).mean()),
        "n": int(diffs.size),
        "n_resamples": int(n),
    }
