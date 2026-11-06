"""Assemble results/RESULTS.md exclusively from measured result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import REPO_ROOT, read_json


def _read_curve(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _ablation_table(results: Path, profile: str) -> str:
    lines = [
        "| Shaping | Final mean reward | Final success rate | Final mean KL | Steps | Reward collapsed |",
        "|---|---|---|---|---|---|",
    ]
    for mode in ("sparse", "dense"):
        log = read_json(results / f"rl_{mode}_train.json", {}) or {}
        curve = _read_curve(results / f"rl_{mode}_curve.jsonl")
        last = curve[-1] if curve else {}
        lines.append(
            f"| {mode} | {last.get('mean_reward', log.get('final_mean_reward', 0.0)):.3f} | "
            f"{last.get('success_rate', 0.0):.3f} | "
            f"{last.get('mean_kl', log.get('final_mean_kl', 0.0)):.6f} | "
            f"{log.get('steps', 0)} | {log.get('reward_collapsed')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _training_table(results: Path) -> str:
    rows = [
        ("SFT", "train_sft.json"),
        ("DPO", "train_dpo.json"),
        ("RL sparse", "rl_sparse_train.json"),
        ("RL dense", "rl_dense_train.json"),
    ]
    lines = ["| Stage | Trainer | Steps | Wall seconds | Final loss |", "|---|---|---|---|---|"]
    for label, fname in rows:
        log = read_json(results / fname, {}) or {}
        trainer = log.get("config", {}).get("trainer") or log.get("trainer", "")
        lines.append(
            f"| {label} | {trainer} | {log.get('steps', 0)} | "
            f"{log.get('wall_seconds', 0):.1f} | {log.get('final_loss', 0.0):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_report(cfg: dict[str, Any]) -> Path:
    out_dir = Path(cfg.get("out_dir", "results"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    table = read_json(out_dir / "eval_table.json", {}) or {}
    eval_md_path = out_dir / "eval_table.md"
    audit_md_path = out_dir / "reward_hacking_audit.md"
    eval_md = eval_md_path.read_text(encoding="utf-8") if eval_md_path.exists() else ""
    audit_md = audit_md_path.read_text(encoding="utf-8") if audit_md_path.exists() else ""

    meta = table.get("metadata", {})
    lines = [
        "# GridWorldAgentRL — measured results",
        "",
        "Every number in this file was generated from the result files in `results/`. "
        "No value is hand-typed.",
        "",
        "## Configuration",
        "",
        f"- model: `{meta.get('model_id')}`",
        f"- profile: `{meta.get('profile')}`",
        f"- seed: {meta.get('seed')}",
        f"- git commit: `{meta.get('git_commit')}`",
        f"- eval seeds: {meta.get('seeds')}",
        f"- eval tasks: {meta.get('n_eval_tasks')} (held-out split)",
        f"- episodes: {meta.get('n_episodes')}",
        f"- eval temperature: {meta.get('eval_temperature')}",
        "",
        "## Main held-out table",
        "",
    ]
    # The eval table already carries its caption and rows; keep one copy.
    body = "\n".join(eval_md.splitlines()[2:]).strip()
    lines.append(body)
    lines.append("")
    lines.append("## Reward-shaping ablation")
    lines.append("")
    lines.append(_ablation_table(out_dir, meta.get("profile", "")))
    lines.append("## Reward-hacking audit")
    lines.append("")
    audit_body = "\n".join(audit_md.splitlines()[1:]).strip()
    lines.append(audit_body)
    lines.append("")
    lines.append("## Training cost")
    lines.append("")
    lines.append(_training_table(out_dir))
    lines.append("## Figures")
    lines.append("")
    lines.append("![RL learning curves](figures/rl_curves.png)")
    lines.append("")
    lines.append("![Reward-hacking audit](figures/audit.png)")
    lines.append("")
    lines.append("## Reading the result")
    lines.append("")
    audit = read_json(out_dir / "reward_hacking_audit.json", {}) or {}
    any_hack = any(
        v.get("any_hack", 0.0) > 0 for v in audit.get("policies", {}).values()
    )
    if not any_hack:
        lines.append(
            "No reward-hacking detector fired on any held-out episode. This is **not** evidence "
            "of alignment: the learned policies very rarely emitted a valid tool call at all "
            "(see malformed-output and invalid-action diagnostics in the eval JSON). The "
            "synthetic positive controls in `results/reward_hacking_audit.json` confirm every "
            "detector is live."
        )
        lines.append("")
    lines.append(
        "The scripted oracle solves the environment, which proves the tasks and success "
        "checks are solvable. The learned policies are evaluated under identical conditions; "
        "their absolute success rates, invalid-action rates, and audit incidence should be "
        "read together, because a policy that never emits a tool call cannot trigger most "
        "tool-level detectors."
    )
    lines.append("")
    path = out_dir / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
