from __future__ import annotations

import pytest

from gwa.metrics import compute_metrics


def _traj(**kw):
    base = {
        "success": False,
        "tool_calls": 0,
        "optimal_steps": 1,
        "tool_failures": 0,
        "invalid_actions": 0,
        "malformed_turns": 0,
        "replans": 0,
        "turns": [{}, {}],
        "terminal_reward": 0.0,
        "dense_return": 0.0,
        "family": "triage_and_route",
    }
    base.update(kw)
    return base


def test_metrics_hand_computed_fixture():
    trajectories = [
        # success, exactly optimal
        _traj(success=True, tool_calls=2, optimal_steps=2, terminal_reward=1.0),
        # success, half efficiency, one tool failure
        _traj(
            success=True,
            tool_calls=4,
            optimal_steps=2,
            tool_failures=1,
            malformed_turns=1,
            terminal_reward=0.9,
        ),
        # failure after one failure and two invalid actions
        _traj(
            success=False,
            tool_calls=3,
            optimal_steps=2,
            tool_failures=1,
            invalid_actions=2,
            malformed_turns=2,
            replans=1,
        ),
    ]
    m = compute_metrics(trajectories)
    assert m["n_episodes"] == 3
    assert m["task_success_rate"] == pytest.approx(2 / 3)
    assert m["plan_step_efficiency"] == pytest.approx((2 / 2 + 2 / 4) / 2)
    assert m["recovery_rate"] == pytest.approx(1 / 2)
    assert m["invalid_action_rate"] == pytest.approx(2 / 9)
    assert m["malformed_output_rate"] == pytest.approx(3 / 6)
    assert m["mean_turns"] == 2.0
    assert m["mean_terminal_reward"] == pytest.approx((1.0 + 0.9 + 0.0) / 3)
    assert m["n_success"] == 2
    assert m["n_recovery_denominator"] == 2


def test_efficiency_refusal_zero_calls_counts_as_one():
    m = compute_metrics([_traj(success=True, optimal_steps=0, tool_calls=0)])
    assert m["plan_step_efficiency"] == 1.0


def test_recovery_empty_denominator_is_zero():
    m = compute_metrics([_traj(success=True, tool_calls=1)])
    assert m["recovery_rate"] == 0.0
    assert m["n_recovery_denominator"] == 0


def test_empty_metrics():
    m = compute_metrics([])
    assert m["n_episodes"] == 0
    assert m["task_success_rate"] == 0.0
