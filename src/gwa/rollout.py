"""Collect rollouts for a policy and persist trajectories plus a metrics summary."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .agent.loop import Trajectory, run_episode
from .agent.policies import Policy
from .env.crm_env import CRMEnv
from .env.reward import RewardConfig
from .env.tasks import Task
from .metrics import compute_metrics, per_family_metrics
from .utils import JsonlWriter, mean, run_metadata, write_json


def make_env_for_task(task: Task, cfg: dict[str, Any], reward_mode: str | None = None) -> CRMEnv:
    env_cfg = cfg.get("env", {})
    return CRMEnv(
        task,
        max_steps=int(env_cfg.get("max_steps", 8)),
        reward_mode=reward_mode or str(env_cfg.get("reward_mode", "dense")),
        reward_cfg=RewardConfig.from_dict(cfg.get("reward")),
        env_sizes=task.env_sizes,
    )


def collect(
    policy: Policy,
    tasks: list[Task],
    n_per_task: int,
    cfg: dict[str, Any],
    tag: str,
    out_dir: str | Path = "results",
    reward_mode: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run ``n_per_task`` episodes per task, writing JSONL and a summary JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / f"rollouts_{tag}.jsonl"
    summary_path = out / f"metrics_{tag}.json"
    started = time.time()
    trajs: list[Trajectory] = []

    total = len(tasks) * n_per_task
    i = 0
    with JsonlWriter(jsonl_path) as writer:
        for task in tasks:
            for rep in range(n_per_task):
                i += 1
                env = make_env_for_task(task, cfg, reward_mode=reward_mode)
                episode_seed = int(cfg.get("seed", 0)) + rep
                traj = run_episode(env, task, policy, cfg, episode_seed=episode_seed)
                writer.write(traj.to_dict())
                trajs.append(traj)
                if verbose:
                    print(
                        f"[rollout {tag}] {i}/{total} {task.task_id} "
                        f"success={traj.success} calls={traj.tool_calls} "
                        f"term={traj.terminal_reward:.3f} {traj.wall_seconds:.2f}s",
                        flush=True,
                    )

    metrics = compute_metrics(trajs)
    summary = {
        "tag": tag,
        "policy": getattr(policy, "name", "unknown"),
        "n_per_task": n_per_task,
        "n_tasks": len(tasks),
        "reward_mode": reward_mode or cfg.get("env", {}).get("reward_mode", "dense"),
        "wall_seconds": round(time.time() - started, 2),
        "mean_wall_seconds_per_episode": round(mean(t.wall_seconds for t in trajs), 3),
        "metrics": metrics,
        "per_family": per_family_metrics(trajs),
        "metadata": run_metadata(cfg),
    }
    write_json(summary_path, summary)
    if verbose:
        print(
            f"[rollout {tag}] success={metrics['task_success_rate']:.3f} "
            f"efficiency={metrics['plan_step_efficiency']:.3f} "
            f"invalid={metrics['invalid_action_rate']:.3f} "
            f"-> {summary_path}",
            flush=True,
        )
    return summary


def load_rollouts(path: str | Path) -> list[dict[str, Any]]:
    import json

    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
