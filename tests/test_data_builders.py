from __future__ import annotations

import json

from gwa.data.build_prefs import build as build_prefs
from gwa.data.build_prefs import load_pairs
from gwa.data.build_sft import build as build_sft
from gwa.data.build_sft import load_sft_records
from gwa.env.tasks import dump_splits, load_tasks
from gwa.utils import REPO_ROOT, load_config


class StubMalformedPolicy:
    """Fast stand-in for the base model: always produces malformed output."""

    name = "stub_malformed"

    def bind_task(self, task) -> None:
        return None

    def act(self, prompt: str) -> str:
        return "I will look at the ticket and then decide what to do."


def _cfg():
    cfg = load_config(profile="smoke")
    cfg["data"]["n_train"] = 12
    cfg["data"]["n_eval"] = 6
    dump_splits(cfg)
    return cfg


def test_sft_dataset_nonempty_and_valid():
    cfg = _cfg()
    summary = build_sft(cfg)
    assert summary["n_records"] > 0
    assert summary["eval_seed_leakage"] is False
    records = load_sft_records(cfg)
    assert len(records) == summary["n_records"]
    train_seeds = {(t.family, t.seed) for t in load_tasks(cfg["data"]["tasks_train"])}
    eval_seeds = {(t.family, t.seed) for t in load_tasks(cfg["data"]["tasks_eval"])}
    for rec in records:
        assert rec["text"].strip()
        assert rec["completion"].strip()
        assert ("ACTION:" in rec["completion"]) or ("FINISH:" in rec["completion"])
        assert rec["kind"] in {"scripted", "recovery_tool_error", "recovery_malformed"}
        assert (rec["family"], rec["seed"]) in train_seeds
        assert (rec["family"], rec["seed"]) not in eval_seeds


def test_dpo_dataset_nonempty_and_valid():
    cfg = _cfg()
    summary = build_prefs(cfg, base_policy=StubMalformedPolicy(), max_pairs=4)
    assert summary["n_pairs"] > 0
    assert summary["eval_seed_leakage"] is False
    pairs = load_pairs(cfg)
    assert 0 < len(pairs) <= 4
    eval_seeds = {(t.family, t.seed) for t in load_tasks(cfg["data"]["tasks_eval"])}
    train = {t.task_id: t for t in load_tasks(cfg["data"]["tasks_train"])}
    for pair in pairs:
        assert set(pair) >= {"prompt", "chosen", "rejected", "task_id", "family", "reason"}
        assert pair["chosen"] != pair["rejected"]
        assert pair["chosen"].strip() and pair["rejected"].strip()
        task = train[pair["task_id"]]
        assert (task.family, task.seed) not in eval_seeds
        assert pair["reason"] in {
            "malformed",
            "invalid_tool_action",
            "lower_terminal_reward",
            "invalid_remedy_recovery",
        }


def test_prefs_header_records_rule():
    cfg = _cfg()
    build_prefs(cfg, base_policy=StubMalformedPolicy(), max_pairs=2)
    path = REPO_ROOT / cfg["data"]["prefs_out"]
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["_type"] == "header"
    assert "chosen" in first["construction_rule"]
    assert "rejected" in first["construction_rule"]
