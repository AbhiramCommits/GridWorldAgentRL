"""Held-out head-to-head evaluation with bootstrap confidence intervals."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .agent.loop import run_episode
from .agent.policies import build_policy
from .env.tasks import load_tasks
from .metrics import compute_metrics, per_family_metrics
from .rollout import make_env_for_task
from .stats import bootstrap_ci, paired_bootstrap_diff
from .utils import (
    REPO_ROOT,
    JsonlWriter,
    run_metadata,
    set_seed,
    write_json,
)


def _episode_efficiency(ep: dict[str, Any]) -> float | None:
    if not ep.get("success"):
        return None
    optimal = int(ep.get("optimal_steps", 0))
    actual = int(ep.get("tool_calls", 0))
    if actual == 0:
        return 1.0 if optimal == 0 else 0.0
    return optimal / actual


def _summarize_policy(episodes: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    m = compute_metrics(episodes)
    n_boot = int(cfg.get("eval", {}).get("n_bootstrap", 10000))
    success = [1.0 if e["success"] else 0.0 for e in episodes]
    eff = [x for x in (_episode_efficiency(e) for e in episodes) if x is not None]
    failures = [e for e in episodes if int(e.get("tool_failures", 0)) >= 1]
    recovery = [1.0 if e["success"] else 0.0 for e in failures]
    invalid_rates = [
        (e.get("invalid_actions", 0) / e["tool_calls"]) if e.get("tool_calls", 0) else 0.0
        for e in episodes
    ]
    return {
        "metrics": m,
        "ci": {
            "task_success_rate": bootstrap_ci(success, n=n_boot),
            "plan_step_efficiency": bootstrap_ci(eff, n=n_boot),
            "recovery_rate": bootstrap_ci(recovery, n=n_boot),
            "invalid_action_rate": bootstrap_ci(invalid_rates, n=n_boot),
        },
        "n_episodes": len(episodes),
        "per_family": per_family_metrics(episodes),
    }


def _paired(
    episodes: dict[str, list[dict[str, Any]]],
    a: str,
    b: str,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if a not in episodes or b not in episodes:
        return None
    ea = {(e["task_id"], e["episode_seed"]): e for e in episodes[a]}
    eb = {(e["task_id"], e["episode_seed"]): e for e in episodes[b]}
    keys = sorted(set(ea) & set(eb))
    if not keys:
        return None
    n_boot = int(cfg.get("eval", {}).get("n_bootstrap", 10000))
    a_succ = [1.0 if ea[k]["success"] else 0.0 for k in keys]
    b_succ = [1.0 if eb[k]["success"] else 0.0 for k in keys]
    a_eff = [_episode_efficiency(ea[k]) or 0.0 for k in keys]
    b_eff = [_episode_efficiency(eb[k]) or 0.0 for k in keys]
    return {
        "a": a,
        "b": b,
        "n_pairs": len(keys),
        "success": paired_bootstrap_diff(a_succ, b_succ, n=n_boot),
        "plan_step_efficiency": paired_bootstrap_diff(a_eff, b_eff, n=n_boot),
    }


def run_benchmark(
    policies: list[str],
    eval_tasks,
    seeds: list[int],
    cfg: dict[str, Any],
    backend: str = "hf",
) -> dict[str, Any]:
    out_dir = Path(cfg.get("out_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "eval_episodes.jsonl"
    all_eps: dict[str, list[dict[str, Any]]] = {}
    started = time.time()

    with JsonlWriter(episodes_path) as writer:
        for policy_name in policies:
            policy = build_policy(
                policy_name,
                cfg,
                backend=backend,
                temperature=float(cfg.get("eval", {}).get("temperature", 0.2))
                if policy_name != "scripted"
                else 0.0,
            )
            eps: list[dict[str, Any]] = []
            for seed in seeds:
                set_seed(int(seed))
                for task in eval_tasks:
                    env = make_env_for_task(task, cfg)
                    traj = run_episode(env, task, policy, cfg, episode_seed=int(seed))
                    rec = traj.to_dict()
                    rec["eval_seed"] = int(seed)
                    writer.write(rec)
                    eps.append(rec)
            all_eps[policy_name] = eps
            m = compute_metrics(eps)
            print(
                f"[eval] {policy_name}: success={m['task_success_rate']:.3f} "
                f"efficiency={m['plan_step_efficiency']:.3f} "
                f"invalid={m['invalid_action_rate']:.3f} "
                f"turns={m['mean_turns']:.2f}",
                flush=True,
            )

    summaries = {name: _summarize_policy(eps, cfg) for name, eps in all_eps.items()}
    pairs = {
        "rl_sparse_vs_base": _paired(all_eps, "rl_sparse", "base", cfg),
        "rl_dense_vs_base": _paired(all_eps, "rl_dense", "base", cfg),
        "rl_sparse_vs_sft": _paired(all_eps, "rl_sparse", "sft", cfg),
        "rl_dense_vs_sft": _paired(all_eps, "rl_dense", "sft", cfg),
        "rl_sparse_vs_dpo": _paired(all_eps, "rl_sparse", "dpo", cfg),
        "rl_dense_vs_dpo": _paired(all_eps, "rl_dense", "dpo", cfg),
        "dense_vs_sparse": _paired(all_eps, "rl_dense", "rl_sparse", cfg),
        "sft_vs_base": _paired(all_eps, "sft", "base", cfg),
        "dpo_vs_base": _paired(all_eps, "dpo", "base", cfg),
    }
    payload = {
        "metadata": run_metadata(
            cfg,
            backend=backend,
            seeds=[int(s) for s in seeds],
            n_eval_tasks=len(eval_tasks),
            n_episodes=sum(len(v) for v in all_eps.values()),
            wall_seconds=round(time.time() - started, 2),
            eval_temperature=cfg.get("eval", {}).get("temperature"),
        ),
        "policies": summaries,
        "paired": pairs,
    }
    write_json(out_dir / "eval_table.json", payload)
    (out_dir / "eval_table.md").write_text(_markdown_table(payload), encoding="utf-8")
    return payload


def _fmt_ci(ci: dict[str, float], scale: float = 1.0, digits: int = 3) -> str:
    return (
        f"{ci['mean'] * scale:.{digits}f} "
        f"[{ci['ci_low'] * scale:.{digits}f}, {ci['ci_high'] * scale:.{digits}f}]"
    )


def _markdown_table(payload: dict[str, Any]) -> str:
    lines = [
        "# Held-out evaluation",
        "",
        f"Model: `{payload['metadata']['model_id']}` | profile: `{payload['metadata']['profile']}` "
        f"| seeds: {payload['metadata']['seeds']} | eval tasks: {payload['metadata']['n_eval_tasks']} "
        f"| episodes: {payload['metadata']['n_episodes']}",
        "",
        "| Policy | Task success (95% CI) | Plan-step efficiency | Recovery rate | "
        "Invalid action rate | Mean turns |",
        "|---|---|---|---|---|---|",
    ]
    for name, summary in payload["policies"].items():
        ci = summary["ci"]
        m = summary["metrics"]
        lines.append(
            f"| {name} | {_fmt_ci(ci['task_success_rate'])} | "
            f"{_fmt_ci(ci['plan_step_efficiency'])} | {_fmt_ci(ci['recovery_rate'])} | "
            f"{_fmt_ci(ci['invalid_action_rate'])} | {m['mean_turns']:.2f} |"
        )
    lines.append("")
    lines.append("## Paired bootstrap comparisons")
    lines.append("")
    lines.append("| Comparison | Δ success [95% CI] | P(a > b) | Δ efficiency [95% CI] |")
    lines.append("|---|---|---|---|")
    for _key, comp in payload["paired"].items():
        if comp is None:
            continue
        s = comp["success"]
        e = comp["plan_step_efficiency"]
        lines.append(
            f"| {comp['a']} − {comp['b']} | "
            f"{s['mean_diff']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] | "
            f"{s['p_a_better']:.3f} | "
            f"{e['mean_diff']:+.3f} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}] |"
        )
    lines.append("")
    return "\n".join(lines)


def run_benchmark_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("data", {})
    eval_path = REPO_ROOT / data_cfg.get("tasks_eval", "data/tasks_eval.jsonl")
    if not eval_path.exists():
        from .env.tasks import dump_splits

        dump_splits(cfg)
    eval_tasks = load_tasks(eval_path)
    policies = list(cfg.get("eval", {}).get("policies", ["base", "sft", "dpo", "scripted"]))
    seeds = [int(s) for s in cfg.get("data", {}).get("eval_seeds", [0])]
    return run_benchmark(
        policies, eval_tasks, seeds, cfg, backend=str(cfg.get("backend", "hf"))
    )
