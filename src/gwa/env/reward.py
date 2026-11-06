"""Verifiable reward functions over the final CRM state.

Two shaping modes share one :class:`RewardConfig`:

* ``terminal`` (sparse): 1.0 on programmatic task success, otherwise 0.0, minus a
  penalty for every invalid tool call made during the episode.
* ``dense``: per-step credit for newly achieved subgoals, a small bonus for a valid
  tool call, and a per-step cost, plus the same terminal reward at episode end.

The environment is the only source of reward: success is computed by programmatic
checks over ``CRMState``, never by a learned reward model or human label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .state import CRMState
    from .tasks import Task


@dataclass
class RewardConfig:
    invalid_action_penalty: float = 0.10
    valid_tool_bonus: float = 0.02
    step_cost: float = 0.01
    subgoal_weight: float = 0.25
    success_bonus: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RewardConfig:
        data = data or {}
        return cls(
            invalid_action_penalty=float(data.get("invalid_action_penalty", 0.10)),
            valid_tool_bonus=float(data.get("valid_tool_bonus", 0.02)),
            step_cost=float(data.get("step_cost", 0.01)),
            subgoal_weight=float(data.get("subgoal_weight", 0.25)),
            success_bonus=float(data.get("success_bonus", 1.0)),
        )


def task_progress(task: Task, state: CRMState) -> tuple[bool, dict[str, bool], float]:
    """Return (success, subgoal flags, fraction of flags satisfied)."""
    ok, flags = task.success_check(state)
    progress = sum(bool(v) for v in flags.values()) / max(1, len(flags))
    return ok, flags, float(progress)


def terminal_reward(
    task: Task, state: CRMState, cfg: RewardConfig | None = None
) -> float:
    """Sparse verifiable reward: success bonus minus invalid-action penalty."""
    cfg = cfg or RewardConfig()
    ok, _ = task.success_check(state)
    return (cfg.success_bonus if ok else 0.0) - cfg.invalid_action_penalty * state.invalid_actions


def dense_reward(
    task: Task,
    state: CRMState,
    info: dict[str, Any],
    step: int,
    cfg: RewardConfig | None = None,
) -> float:
    """Dense per-step reward: new-subgoal credit + valid-tool bonus - step cost."""
    cfg = cfg or RewardConfig()
    prev = float(info.get("prev_progress", 0.0))
    now = float(info.get("progress", 0.0))
    new_credit = max(0.0, now - prev)
    valid_bonus = cfg.valid_tool_bonus if info.get("tool_ok") else 0.0
    return cfg.subgoal_weight * new_credit + valid_bonus - cfg.step_cost


def episode_return(mode: str, step_rewards: list[float], terminal: float) -> float:
    """Combine per-step shaped rewards with the terminal reward for one episode."""
    if mode == "sparse":
        return float(terminal)
    if mode == "dense":
        return float(sum(step_rewards) + terminal)
    raise ValueError(f"unknown reward mode {mode!r}; expected 'sparse' or 'dense'")
