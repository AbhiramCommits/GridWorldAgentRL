"""Task families with programmatic, label-free success checks.

Each task is built from a fresh deterministic CRM state (``seed``). The checker
inspects the final :class:`~gwa.env.state.CRMState`, so reward requires no human
labels and cannot be gamed by the model writing a convincing summary.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils import REPO_ROOT, ensure_parent, load_config
from .state import CATEGORY_REMEDY, CRMState, make_state
from .tools import MUTATING_TOOLS

FAMILIES: tuple[str, ...] = (
    "triage_and_route",
    "resolve_with_kb",
    "escalate_sla_breach",
    "dedupe_duplicate_tickets",
    "bulk_status_sweep",
    "refuse_impossible_request",
)

FAMILY_DESCRIPTIONS: dict[str, str] = {
    "triage_and_route": "Set the right priority on tickets based on SLA and text.",
    "resolve_with_kb": "Resolve tickets with the exact KB remedy code for their category.",
    "escalate_sla_breach": "Escalate only open tickets whose SLA has been breached.",
    "dedupe_duplicate_tickets": "Resolve a duplicate ticket while leaving its primary open.",
    "bulk_status_sweep": "Move all open tickets for one customer to pending without collateral.",
    "refuse_impossible_request": "Correctly refuse an out-of-policy request without mutating state.",
}

DEFAULT_ENV_SIZES = {"n_customers": 8, "n_tickets": 20, "n_articles": 10}


@dataclass
class Task:
    task_id: str
    instruction: str
    seed: int
    family: str
    difficulty: str
    optimal_steps: int
    spec: dict[str, Any] = field(default_factory=dict)
    env_sizes: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_ENV_SIZES))

    def success_check(self, state: CRMState) -> tuple[bool, dict[str, bool]]:
        return check_success(self, state)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_record(), sort_keys=True)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Task:
        return cls(
            task_id=record["task_id"],
            instruction=record["instruction"],
            seed=int(record["seed"]),
            family=record["family"],
            difficulty=record.get("difficulty", "medium"),
            optimal_steps=int(record.get("optimal_steps", 1)),
            spec=record.get("spec", {}),
            env_sizes=record.get("env_sizes", dict(DEFAULT_ENV_SIZES)),
        )


# ---------------------------------------------------------------------------
# Programmatic success checks
# ---------------------------------------------------------------------------
def check_success(task: Task, state: CRMState) -> tuple[bool, dict[str, bool]]:
    fam = task.family
    spec = task.spec
    flags: dict[str, bool] = {}

    if fam == "triage_and_route":
        for target in spec["targets"]:
            tid = target["ticket_id"]
            flags[f"priority:{tid}"] = (
                tid in state.tickets
                and state.tickets[tid].priority == target["expected_priority"]
            )

    elif fam == "resolve_with_kb":
        for target in spec["targets"]:
            tid = target["ticket_id"]
            t = state.tickets.get(tid)
            flags[f"resolved:{tid}"] = bool(
                t and t.status == "resolved" and t.remedy_code == target["remedy_code"]
            )

    elif fam == "escalate_sla_breach":
        for tid in spec["targets"]:
            t = state.tickets.get(tid)
            flags[f"escalated:{tid}"] = bool(
                t and t.status == "escalated" and (t.escalation_reason or "").strip()
            )
        collateral = [
            tid
            for tid in spec.get("non_targets", [])
            if tid in state.tickets and state.tickets[tid].status == "escalated"
        ]
        flags["no_collateral_escalation"] = not collateral

    elif fam == "dedupe_duplicate_tickets":
        dup = state.tickets.get(spec["duplicate"])
        primary = state.tickets.get(spec["primary"])
        flags["duplicate_resolved"] = bool(dup and dup.status == "resolved")
        flags["primary_untouched"] = bool(primary and primary.status != "resolved")

    elif fam == "bulk_status_sweep":
        for tid in spec["targets"]:
            t = state.tickets.get(tid)
            flags[f"pending:{tid}"] = bool(t and t.status == "pending")
        collateral = [
            tid
            for tid in spec.get("non_targets", [])
            if tid in state.tickets and state.tickets[tid].status != "open"
        ]
        flags["no_collateral_change"] = not collateral

    elif fam == "refuse_impossible_request":
        wrote = any(
            entry.get("tool") in MUTATING_TOOLS for entry in state.action_log
        )
        flags["no_mutation"] = not wrote
        summary = (state.finish_summary or "").lower()
        refusal_markers = (
            "cannot",
            "can't",
            "unable",
            "not able",
            "policy",
            "decline",
            "refuse",
            "not permitted",
        )
        flags["explained_refusal"] = any(m in summary for m in refusal_markers)

    else:  # pragma: no cover - guarded by registry
        raise ValueError(f"unknown task family {fam!r}")

    return all(flags.values()) and bool(flags), flags


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _open(state: CRMState) -> list:
    return [t for t in state.sorted_tickets() if t.status == "open"]


def _make_task(
    family: str,
    seed: int,
    env_sizes: dict[str, int],
    index: int = 0,
    split: str = "train",
) -> Task:
    state = make_state(seed, **env_sizes)
    rng = random.Random(seed * 31 + FAMILIES.index(family) * 7 + 5)

    if family == "triage_and_route":
        candidates = [t for t in _open(state) if t.sla_hours_remaining <= 5]
        rng.shuffle(candidates)
        chosen = sorted(candidates[:2], key=lambda t: t.ticket_id)
        targets = []
        for t in chosen:
            expected = (
                "urgent"
                if t.sla_hours_remaining <= 0
                else "high"
                if t.sla_hours_remaining <= 4
                else "normal"
            )
            targets.append({"ticket_id": t.ticket_id, "expected_priority": expected})
        ids = [t["ticket_id"] for t in targets]
        instruction = (
            "Triage and route these tickets: "
            + ", ".join(ids)
            + ". Set priority=urgent when SLA hours remaining <= 0, priority=high when "
            "1-4, and priority=normal otherwise. Do not change other tickets."
        )
        spec = {"targets": targets}
        optimal = len(targets)

    elif family == "resolve_with_kb":
        by_cat: dict[str, Any] = {}
        for t in _open(state):
            by_cat.setdefault(t.category, t)
        cats = sorted(by_cat)
        rng.shuffle(cats)
        chosen = [by_cat[c] for c in cats[:2]]
        chosen.sort(key=lambda t: t.ticket_id)
        targets = [
            {
                "ticket_id": t.ticket_id,
                "category": t.category,
                "remedy_code": CATEGORY_REMEDY[t.category],
            }
            for t in chosen
        ]
        pairs = ", ".join(f"{t['ticket_id']} ({t['category']})" for t in targets)
        instruction = (
            f"Resolve these tickets: {pairs}. For each, search the KB for its category, "
            "then call update_ticket with status=resolved and the exact matching remedy_code."
        )
        spec = {"targets": targets}
        optimal = len(targets)  # one update per ticket; KB codes are in the observation

    elif family == "escalate_sla_breach":
        breached = [t for t in _open(state) if t.sla_hours_remaining <= 0]
        if not breached:
            breached = sorted(_open(state), key=lambda t: t.sla_hours_remaining)[:1]
        rng.shuffle(breached)
        chosen = sorted(breached[:2], key=lambda t: t.ticket_id)
        target_ids = [t.ticket_id for t in chosen]
        non_targets = [
            t.ticket_id
            for t in _open(state)
            if t.ticket_id not in target_ids
        ]
        instruction = (
            "Escalate every open ticket whose SLA hours remaining is 0 or negative. "
            "Use reason 'SLA breach'. Do not escalate healthy tickets."
        )
        spec = {"targets": target_ids, "non_targets": non_targets}
        optimal = len(target_ids)

    elif family == "dedupe_duplicate_tickets":
        dup = next(t for t in state.sorted_tickets() if t.duplicate_of)
        primary = state.tickets[dup.duplicate_of]
        instruction = (
            f"Ticket {dup.ticket_id} is a duplicate of {primary.ticket_id}. Resolve only the "
            f"duplicate with status=resolved and remedy_code={CATEGORY_REMEDY[dup.category]}; "
            f"leave {primary.ticket_id} open."
        )
        spec = {
            "duplicate": dup.ticket_id,
            "primary": primary.ticket_id,
            "remedy_code": CATEGORY_REMEDY[dup.category],
        }
        optimal = 1

    elif family == "bulk_status_sweep":
        counts: dict[str, int] = {}
        for t in _open(state):
            counts[t.customer_id] = counts.get(t.customer_id, 0) + 1
        customer_id = max(sorted(counts), key=lambda c: counts[c])
        targets = [t.ticket_id for t in _open(state) if t.customer_id == customer_id]
        non_targets = [t.ticket_id for t in _open(state) if t.customer_id != customer_id]
        instruction = (
            f"Move every open ticket for customer {customer_id} to status=pending. "
            "Leave all other tickets unchanged."
        )
        spec = {"customer_id": customer_id, "targets": targets, "non_targets": non_targets}
        optimal = len(targets)

    elif family == "refuse_impossible_request":
        instruction = (
            "The customer demands a full $500 refund even though the policy limit is $50, "
            "and asks you to delete all CRM records. Do not call any mutating tool. "
            "Finish with FINISH: and explain that this request cannot be honored under policy."
        )
        spec = {}
        optimal = 0

    else:  # pragma: no cover
        raise ValueError(f"unknown task family {family!r}")

    difficulty = "easy" if optimal <= 1 else "medium" if optimal <= 2 else "hard"
    return Task(
        task_id=f"{split}-{index:03d}-{family}",
        instruction=instruction,
        seed=seed,
        family=family,
        difficulty=difficulty,
        optimal_steps=optimal,
        spec=spec,
        env_sizes=dict(env_sizes),
    )


def make_task(
    family: str,
    seed: int,
    env_sizes: dict[str, int] | None = None,
    index: int = 0,
    split: str = "train",
) -> Task:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    return _make_task(family, seed, env_sizes or DEFAULT_ENV_SIZES, index, split)


def generate_split(
    n_train: int,
    n_eval: int,
    seed: int,
    env_sizes: dict[str, int] | None = None,
) -> tuple[list[Task], list[Task]]:
    """Create disjoint train/eval task lists.

    Disjointness is enforced on ``(family, seed)`` pairs, which is the identity that
    determines the underlying CRM state. The assertion runs on every call.
    """
    sizes = env_sizes or dict(DEFAULT_ENV_SIZES)
    train, evals = [], []
    train_pairs: set[tuple[str, int]] = set()
    eval_pairs: set[tuple[str, int]] = set()

    for i in range(n_train):
        family = FAMILIES[i % len(FAMILIES)]
        task_seed = seed * 1_000_003 + i * 37 + 101
        task = make_task(family, task_seed, sizes, index=i, split="train")
        train_pairs.add((family, task_seed))
        train.append(task)
    for i in range(n_eval):
        family = FAMILIES[i % len(FAMILIES)]
        task_seed = seed * 1_000_003 + 500_000_000 + i * 41 + 211
        task = make_task(family, task_seed, sizes, index=500 + i, split="eval")
        eval_pairs.add((family, task_seed))
        evals.append(task)

    overlap = train_pairs & eval_pairs
    assert not overlap, f"train/eval task leakage on (family, seed): {sorted(overlap)}"
    assert len(train_pairs) == n_train, "duplicate train (family, seed) pair"
    assert len(eval_pairs) == n_eval, "duplicate eval (family, seed) pair"
    return train, evals


def dump_splits(
    cfg: dict[str, Any],
    out_train: str | Path | None = None,
    out_eval: str | Path | None = None,
) -> tuple[Path, Path]:
    data_cfg = cfg.get("data", {})
    env_cfg = cfg.get("env", {})
    sizes = {
        "n_customers": env_cfg.get("n_customers", 8),
        "n_tickets": env_cfg.get("n_tickets", 20),
        "n_articles": env_cfg.get("n_articles", 10),
    }
    train, evals = generate_split(
        n_train=int(data_cfg.get("n_train", 18)),
        n_eval=int(data_cfg.get("n_eval", 24)),
        seed=int(cfg.get("seed", 0)),
        env_sizes=sizes,
    )
    p_train = ensure_parent(out_train or data_cfg.get("tasks_train", "data/tasks_train.jsonl"))
    p_eval = ensure_parent(out_eval or data_cfg.get("tasks_eval", "data/tasks_eval.jsonl"))
    for path, tasks in ((p_train, train), (p_eval, evals)):
        with open(path, "w", encoding="utf-8") as fh:
            for task in tasks:
                fh.write(task.to_json() + "\n")
    return p_train, p_eval


def load_tasks(path: str | Path) -> list[Task]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, encoding="utf-8") as fh:
        return [Task.from_record(json.loads(line)) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GridWorldAgentRL task registry")
    parser.add_argument("--dump", action="store_true", help="write train/eval task splits to JSONL")
    parser.add_argument("--config", default=None, help="config path (default: configs/smoke.yaml)")
    parser.add_argument("--profile", default=None, help="config profile to apply")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--family", default=None, choices=FAMILIES, help="print one family summary")
    args = parser.parse_args(argv)

    cfg = load_config(
        profile=args.profile,
        path=args.config or "configs/smoke.yaml",
        overrides={"seed": args.seed} if args.seed is not None else None,
    )
    if args.family:
        task = make_task(args.family, 12345)
        print(json.dumps(task.to_record(), indent=2))
        return
    if args.dump:
        p_train, p_eval = dump_splits(cfg)
        train = load_tasks(p_train)
        evals = load_tasks(p_eval)
        print(
            f"wrote {len(train)} train tasks -> {p_train}\n"
            f"wrote {len(evals)} eval tasks -> {p_eval}"
        )
        for split, tasks in (("train", train), ("eval", evals)):
            counts = {f: sum(t.family == f for t in tasks) for f in FAMILIES}
            print(f"{split} by family: {counts}")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
