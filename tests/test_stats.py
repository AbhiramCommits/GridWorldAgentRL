from __future__ import annotations

import numpy as np
import pytest

from gwa.stats import bootstrap_ci, paired_bootstrap_diff


def test_bootstrap_ci_known_bernoulli():
    # Analytic mean is 0.5; the normal-approximation 95% CI is 0.5 +/- 1.96*sqrt(.25/2000).
    values = [0.0] * 1000 + [1.0] * 1000
    ci = bootstrap_ci(values, n=4000, seed=0)
    assert ci["mean"] == pytest.approx(0.5)
    assert 0.47 < ci["ci_low"] < 0.49
    assert 0.51 < ci["ci_high"] < 0.53
    assert ci["n"] == 2000


def test_bootstrap_ci_constant_is_exact():
    ci = bootstrap_ci([0.25] * 50, n=500, seed=1)
    assert ci["mean"] == ci["ci_low"] == ci["ci_high"] == 0.25


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([], n=100)
    assert ci["mean"] == 0.0 and ci["n"] == 0


def test_paired_bootstrap_diff_all_positive():
    out = paired_bootstrap_diff([1, 2, 3, 4], [0, 0, 0, 0], n=1000, seed=0)
    assert out["mean_diff"] == pytest.approx(2.5)
    assert out["ci_low"] > 0
    assert out["ci_high"] >= out["mean_diff"]
    assert out["p_a_better"] == 1.0


def test_paired_bootstrap_diff_zero():
    out = paired_bootstrap_diff([1, 2, 3], [1, 2, 3], n=1000, seed=0)
    assert out["mean_diff"] == 0.0
    assert out["ci_low"] == out["ci_high"] == 0.0
    assert out["p_a_better"] == 0.0


def test_paired_bootstrap_requires_equal_length():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 2], [1])


def test_bootstrap_matches_numpy_resample_shape():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    ci = bootstrap_ci(values, n=2000, seed=2)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
