"""Plan -> execute -> observe -> replan agent loop and trajectory records."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..env.crm_env import CRMEnv
from ..env.reward import episode_return
from ..env.tasks import Task
from ..env.tools import ToolCall
from ..utils import set_seed
from .parser import ParsedAction, parse_response
from .policies import Policy
from .prompt import build_user_content


@dataclass
class Trajectory:
    task_id: str
    family: str
    policy: str
    seed: int
    episode_seed: int
    turns: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    rewards: list[float] = field(default_factory=list)
    plans: list[str] = field(default_factory=list)
    plan_revisions: int = 0
    replans: int = 0
    tool_failures: int = 0
    malformed_turns: int = 0
    invalid_actions: int = 0
    success: bool = False
    terminal_reward: float = 0.0
    dense_return: float = 0.0
    finish_summary: str | None = None
    initial_hash: str | None = None
    final_hash: str | None = None
    wall_seconds: float = 0.0
    max_steps_hit: bool = False
    error: str | None = None
    optimal_steps: int = 0
    subgoals: dict[str, bool] = field(default_factory=dict)

    @property
    def actual_tool_calls(self) -> int:
        return self.tool_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "policy": self.policy,
            "seed": self.seed,
            "episode_seed": self.episode_seed,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "rewards": self.rewards,
            "plans": self.plans,
            "plan_revisions": self.plan_revisions,
            "replans": self.replans,
            "tool_failures": self.tool_failures,
            "malformed_turns": self.malformed_turns,
            "invalid_actions": self.invalid_actions,
            "success": self.success,
            "terminal_reward": self.terminal_reward,
            "dense_return": self.dense_return,
            "finish_summary": self.finish_summary,
            "initial_hash": self.initial_hash,
            "final_hash": self.final_hash,
            "wall_seconds": self.wall_seconds,
            "max_steps_hit": self.max_steps_hit,
            "error": self.error,
            "optimal_steps": self.optimal_steps,
            "subgoals": self.subgoals,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True)


def _turn_record(
    turn: int,
    prompt: str,
    response: str,
    parsed: ParsedAction,
    reward: float = 0.0,
    tool_ok: bool | None = None,
    error: str | None = None,
    observation: str = "",
    replanned: bool = False,
) -> dict[str, Any]:
    return {
        "turn": turn,
        "prompt": prompt[-4000:],
        "response": response.strip()[:3000],
        "plan": parsed.plan,
        "action": parsed.action.to_dict() if parsed.action else None,
        "finish": parsed.finish,
        "malformed": parsed.malformed,
        "error": error or parsed.error,
        "tool_ok": tool_ok,
        "reward": reward,
        "replanned": replanned,
        "observation": observation[-4000:],
    }


def run_episode(
    env: CRMEnv,
    task: Task,
    policy: Policy,
    cfg: dict[str, Any],
    episode_seed: int | None = None,
) -> Trajectory:
    """Run one full episode and return a serializable :class:`Trajectory`.

    Replanning is counted when the model emits a *changed* plan on the turn
    immediately after a tool failure or malformed output. A plan that is merely
    repeated does not count.
    """
    started = time.time()
    agent_cfg = cfg.get("agent", {})
    max_turns = int(
        cfg.get("rollout", {}).get("max_turns", agent_cfg.get("max_turns", 8))
    )
    ep_seed = int(episode_seed if episode_seed is not None else cfg.get("seed", 0))
    set_seed(ep_seed)

    if hasattr(policy, "bind_task"):
        policy.bind_task(task)

    obs = env.reset()
    traj = Trajectory(
        task_id=task.task_id,
        family=task.family,
        policy=getattr(policy, "name", type(policy).__name__),
        seed=int(cfg.get("seed", 0)),
        episode_seed=ep_seed,
        initial_hash=env.initial_hash,
        optimal_steps=task.optimal_steps,
    )
    history: list[dict[str, Any]] = []
    last_plan: str | None = None
    pending_failure = False
    consecutive_malformed = 0
    malformed_limit = int(agent_cfg.get("max_consecutive_malformed", 3))
    terminated = False
    truncated = False

    for loop_turn in range(1, max_turns + 1):
        if env.done:
            break
        prompt = build_user_content(task.instruction, obs, history, cfg)
        response = policy.act(prompt)
        parsed = parse_response(response)
        if parsed.plan:
            traj.plans.append(parsed.plan)

        # --- malformed turn: feed the error back, do not advance env time ---
        if parsed.malformed or (parsed.action is None and not parsed.is_finish):
            traj.malformed_turns += 1
            error = parsed.error or "response did not contain a valid ACTION or FINISH"
            replanned = False
            if pending_failure and parsed.plan and parsed.plan != last_plan:
                traj.replans += 1
                traj.plan_revisions += 1
                replanned = True
            traj.turns.append(
                _turn_record(
                    loop_turn, prompt, response, parsed, tool_ok=False, error=error,
                    observation=obs, replanned=replanned,
                )
            )
            history.append(
                {
                    "turn": loop_turn,
                    "assistant": response,
                    "tool": None,
                    "ok": False,
                    "error": error,
                }
            )
            last_plan = parsed.plan or last_plan
            pending_failure = True
            consecutive_malformed += 1
            if hasattr(policy, "observe_tool_result"):
                policy.observe_tool_result(False, error)
            if consecutive_malformed >= malformed_limit:
                traj.error = f"stopped after {consecutive_malformed} consecutive malformed turns"
                break
            continue

        # --- FINISH ---
        if parsed.is_finish:
            env.finish(parsed.finish or "")
            replanned = False
            if pending_failure and parsed.plan and parsed.plan != last_plan:
                traj.replans += 1
                traj.plan_revisions += 1
                replanned = True
            traj.turns.append(
                _turn_record(
                    loop_turn, prompt, response, parsed, tool_ok=True, observation=obs,
                    replanned=replanned,
                )
            )
            terminated = True
            break

        # --- ACTION ---
        action: ToolCall = parsed.action  # type: ignore[assignment]
        replanned = False
        if pending_failure and parsed.plan and parsed.plan != last_plan:
            traj.replans += 1
            traj.plan_revisions += 1
            replanned = True
        last_plan = parsed.plan or last_plan

        new_obs, reward, terminated, truncated, info = env.step(action)
        consecutive_malformed = 0
        traj.tool_calls += 1
        traj.rewards.append(float(reward))
        traj.invalid_actions += int(bool(info.get("invalid_action")))
        if not info.get("tool_ok"):
            traj.tool_failures += 1
            pending_failure = True
        else:
            pending_failure = False
        traj.turns.append(
            _turn_record(
                loop_turn,
                prompt,
                response,
                parsed,
                reward=float(reward),
                tool_ok=bool(info.get("tool_ok")),
                error=info.get("error"),
                observation=new_obs,
                replanned=replanned,
            )
        )
        history.append(
            {
                "turn": loop_turn,
                "assistant": response,
                "tool": action.tool,
                "ok": bool(info.get("tool_ok")),
                "error": info.get("error"),
            }
        )
        if hasattr(policy, "observe_tool_result"):
            policy.observe_tool_result(bool(info.get("tool_ok")), info.get("error"))
        obs = new_obs
        if terminated or truncated:
            break

    ok, subgoals = env.success()
    traj.success = bool(ok)
    traj.subgoals = subgoals
    traj.terminal_reward = float(env.final_terminal_reward())
    traj.dense_return = float(episode_return("dense", traj.rewards, traj.terminal_reward))
    traj.finish_summary = env.state.finish_summary if env.state else None
    traj.final_hash = env.state.state_hash() if env.state else None
    traj.max_steps_hit = bool(truncated)
    if env.state is not None:
        traj.invalid_actions = int(env.state.invalid_actions)
    traj.wall_seconds = round(time.time() - started, 3)
    return traj


def replay_actions(
    env: CRMEnv, task: Task, actions: list[ToolCall], cfg: dict[str, Any]
) -> Trajectory:
    """Deterministic helper used by tests: execute a fixed action list."""
    traj = Trajectory(
        task_id=task.task_id,
        family=task.family,
        policy="replay",
        seed=int(cfg.get("seed", 0)),
        episode_seed=int(cfg.get("seed", 0)),
        optimal_steps=task.optimal_steps,
    )
    env.reset()
    for action in actions:
        if env.done:
            break
        _, reward, terminated, truncated, info = env.step(action)
        traj.tool_calls += 1
        traj.rewards.append(float(reward))
        traj.invalid_actions += int(bool(info.get("invalid_action")))
        traj.tool_failures += int(not info.get("tool_ok"))
        if terminated or truncated:
            break
    ok, flags = env.success()
    traj.success = bool(ok)
    traj.subgoals = flags
    traj.terminal_reward = float(env.final_terminal_reward())
    traj.dense_return = float(episode_return("dense", traj.rewards, traj.terminal_reward))
    traj.final_hash = env.state.state_hash() if env.state else None
    traj.initial_hash = env.initial_hash
    return traj
