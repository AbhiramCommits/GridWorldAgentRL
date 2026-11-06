"""Command-line interface for GridWorldAgentRL.

Subcommands
-----------
rollout      Collect trajectories for one policy and write metrics.
build-sft    Build SFT demonstrations from scripted train rollouts.
build-prefs  Build DPO preference pairs from train rollouts.
train-sft    LoRA SFT with TRL SFTTrainer.
train-dpo    LoRA DPO with TRL DPOTrainer.
train-rl     Online GRPO with the environment as the reward function.
eval         Head-to-head held-out benchmark with bootstrap CIs.
report       Assemble results/RESULTS.md from measured result files.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any

from .env.tasks import dump_splits, load_tasks
from .utils import REPO_ROOT, load_config, set_seed


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=None, help="config profile: smoke, default, large")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"], help="rollout backend")
    parser.add_argument("--config", default=None, help="config YAML (default: configs/smoke.yaml)")
    parser.add_argument("--out", default="results", help="output directory (default: results)")


def _cfg(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "backend", None) is not None:
        overrides.setdefault("backend", args.backend)
    cfg = load_config(
        profile=getattr(args, "profile", None),
        path=getattr(args, "config", None) or "configs/smoke.yaml",
        overrides=overrides or None,
    )
    cfg["out_dir"] = getattr(args, "out", "results")
    return cfg


def _ensure_tasks(cfg: dict[str, Any], split: str):
    key = "tasks_train" if split == "train" else "tasks_eval"
    path = cfg.get("data", {}).get(key, f"data/tasks_{split}.jsonl")
    full = REPO_ROOT / path
    if not full.exists():
        dump_splits(cfg)
    return load_tasks(full)


def cmd_rollout(args: argparse.Namespace) -> int:
    from .agent.policies import build_policy
    from .rollout import collect

    cfg = _cfg(args)
    set_seed(int(cfg.get("seed", 0)))
    tasks = _ensure_tasks(cfg, args.split)
    policy = build_policy(args.policy, cfg, backend=args.backend)
    tag = f"{args.policy}_{args.split}"
    summary = collect(
        policy,
        tasks,
        n_per_task=args.n_per_task,
        cfg=cfg,
        tag=tag,
        out_dir=cfg["out_dir"],
    )
    m = summary["metrics"]
    print(
        f"policy={args.policy} split={args.split} n={m['n_episodes']} "
        f"success={m['task_success_rate']:.3f} efficiency={m['plan_step_efficiency']:.3f} "
        f"recovery={m['recovery_rate']:.3f} invalid={m['invalid_action_rate']:.3f} "
        f"malformed={m['malformed_output_rate']:.3f} turns={m['mean_turns']:.2f}"
    )
    return 0


def _call_optional(module_name: str, func_name: str, *args, **kwargs):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only during incremental builds
        raise SystemExit(f"{module_name} is unavailable: {exc}") from exc
    func = getattr(module, func_name, None)
    if func is None:  # pragma: no cover
        raise SystemExit(f"{module_name}.{func_name} is not implemented")
    return func(*args, **kwargs)


def cmd_build_sft(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    set_seed(int(cfg.get("seed", 0)))
    summary = _call_optional("gwa.data.build_sft", "build", cfg)
    print(f"SFT dataset: {summary}")
    return 0


def cmd_build_prefs(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    set_seed(int(cfg.get("seed", 0)))
    summary = _call_optional("gwa.data.build_prefs", "build", cfg)
    print(f"DPO preferences: {summary}")
    return 0


def cmd_train_sft(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    summary = _call_optional("gwa.train.sft", "run", cfg)
    print(f"SFT training: {summary}")
    return 0


def cmd_train_dpo(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    summary = _call_optional("gwa.train.dpo", "run", cfg)
    print(f"DPO training: {summary}")
    return 0


def cmd_train_rl(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    overrides: dict[str, Any] = {}
    if args.steps is not None:
        overrides["max_steps"] = args.steps
    if args.group_size is not None:
        overrides["group_size"] = args.group_size
    if overrides:
        cfg["rl"] = {**cfg.get("rl", {}), **overrides}
    summary = _call_optional("gwa.train.rl", "run", cfg, args.mode)
    print(f"RL ({args.mode}): {summary}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    summary = _call_optional("gwa.eval", "run_benchmark_from_config", cfg)
    print(f"eval: {summary}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    path = _call_optional("gwa.report", "build_report", cfg)
    print(f"report: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gwa",
        description="Online RL post-training for a multi-turn tool-using CRM agent.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("rollout", help="collect trajectories for one policy")
    p.add_argument(
        "--policy",
        required=True,
        choices=["scripted", "base", "sft", "dpo", "rl_sparse", "rl_dense"],
    )
    p.add_argument("--split", default="eval", choices=["train", "eval"])
    p.add_argument("--n-per-task", type=int, default=1)
    _add_common(p)
    p.set_defaults(func=cmd_rollout)

    p = sub.add_parser("build-sft", help="build SFT demonstrations from scripted train rollouts")
    _add_common(p)
    p.set_defaults(func=cmd_build_sft)

    p = sub.add_parser("build-prefs", help="build DPO preference pairs from train rollouts")
    _add_common(p)
    p.set_defaults(func=cmd_build_prefs)

    p = sub.add_parser("train-sft", help="LoRA SFT with TRL SFTTrainer")
    _add_common(p)
    p.set_defaults(func=cmd_train_sft)

    p = sub.add_parser("train-dpo", help="LoRA DPO with TRL DPOTrainer")
    _add_common(p)
    p.set_defaults(func=cmd_train_dpo)

    p = sub.add_parser("train-rl", help="online GRPO with the environment as reward")
    p.add_argument("--mode", required=True, choices=["sparse", "dense"])
    p.add_argument("--steps", type=int, default=None, help="override rl.max_steps")
    p.add_argument("--group-size", type=int, default=None, help="override rl.group_size")
    _add_common(p)
    p.set_defaults(func=cmd_train_rl)

    p = sub.add_parser("eval", help="head-to-head held-out benchmark with bootstrap CIs")
    _add_common(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("report", help="assemble results/RESULTS.md from result files")
    p.add_argument("--profile", default=None, help="config profile")
    p.add_argument("--config", default=None, help="config YAML")
    p.add_argument("--out", default="results", help="results directory")
    p.set_defaults(func=cmd_report, seed=None, backend="hf")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
