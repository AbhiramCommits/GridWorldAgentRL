"""Trajectory metrics, deliberately separate from task success.

Definitions (all reported explicitly in results):

* ``task_success_rate``: fraction of episodes whose programmatic check passed.
* ``plan_step_efficiency``: mean ``optimal_steps / actual_tool_calls`` over successful
  episodes (0/0 counts as 1.0, a correct no-op refusal).
* ``recovery_rate``: successful episodes / episodes with at least one tool failure.
* ``invalid_action_rate``: invalid tool calls / all tool calls.
* ``malformed_output_rate``: malformed model turns / all model turns.
* ``mean_turns``: mean model turns per episode (includes FINISH and malformed turns).
* ``mean_terminal_reward``: mean sparse verifiable reward.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _as_dict(traj: Any) -> dict[str, Any]:
    return traj.to_dict() if hasattr(traj, "to_dict") else dict(traj)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_metrics(trajectories: Iterable[Any]) -> dict[str, Any]:
    trs = [_as_dict(t) for t in trajectories]
    n = len(trs)
    if n == 0:
        return {
            "n_episodes": 0,
            "task_success_rate": 0.0,
            "plan_step_efficiency": 0.0,
            "recovery_rate": 0.0,
            "invalid_action_rate": 0.0,
            "malformed_output_rate": 0.0,
            "mean_turns": 0.0,
            "mean_terminal_reward": 0.0,
            "mean_dense_return": 0.0,
            "mean_tool_calls": 0.0,
            "replan_rate": 0.0,
            "n_success": 0,
            "n_tool_failures": 0,
            "n_recovery_denominator": 0,
            "n_invalid_actions": 0,
            "n_turns": 0,
            "n_malformed_turns": 0,
            "n_tool_calls": 0,
            "n_replans": 0,
        }

    successes = [t for t in trs if t.get("success")]
    failures = [t for t in trs if int(t.get("tool_failures", 0)) >= 1]
    recovery_successes = [t for t in failures if t.get("success")]
    total_turns = sum(len(t.get("turns", [])) for t in trs)
    total_tool_calls = sum(int(t.get("tool_calls", 0)) for t in trs)
    invalid = sum(int(t.get("invalid_actions", 0)) for t in trs)
    malformed = sum(int(t.get("malformed_turns", 0)) for t in trs)

    efficiencies: list[float] = []
    for t in successes:
        optimal = int(t.get("optimal_steps", 0))
        actual = int(t.get("tool_calls", 0))
        if actual == 0:
            efficiencies.append(1.0 if optimal == 0 else 0.0)
        else:
            efficiencies.append(optimal / actual)

    return {
        "n_episodes": n,
        "task_success_rate": _safe_div(len(successes), n),
        "plan_step_efficiency": _safe_div(sum(efficiencies), len(efficiencies)),
        "recovery_rate": _safe_div(len(recovery_successes), len(failures)),
        "invalid_action_rate": _safe_div(invalid, total_tool_calls),
        "malformed_output_rate": _safe_div(malformed, total_turns),
        "mean_turns": _safe_div(total_turns, n),
        "mean_terminal_reward": _safe_div(
            sum(float(t.get("terminal_reward", 0.0)) for t in trs), n
        ),
        "mean_dense_return": _safe_div(sum(float(t.get("dense_return", 0.0)) for t in trs), n),
        "mean_tool_calls": _safe_div(total_tool_calls, n),
        "replan_rate": _safe_div(sum(1 for t in trs if int(t.get("replans", 0)) > 0), n),
        "n_success": len(successes),
        "n_tool_failures": len(failures),
        "n_recovery_denominator": len(failures),
        "n_invalid_actions": invalid,
        "n_turns": total_turns,
        "n_malformed_turns": malformed,
        "n_tool_calls": total_tool_calls,
        "n_replans": sum(int(t.get("replans", 0)) for t in trs),
    }


def per_family_metrics(trajectories: Iterable[Any]) -> dict[str, dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for traj in trajectories:
        rec = _as_dict(traj)
        by_family.setdefault(rec.get("family", "unknown"), []).append(rec)
    return {family: compute_metrics(recs) for family, recs in sorted(by_family.items())}


METRIC_NAMES: tuple[str, ...] = (
    "task_success_rate",
    "plan_step_efficiency",
    "recovery_rate",
    "invalid_action_rate",
    "malformed_output_rate",
    "mean_turns",
    "mean_terminal_reward",
)
