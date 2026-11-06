from __future__ import annotations

import json
import math

import pytest

from gwa.utils import load_config


@pytest.mark.slow
def test_train_rl_two_steps_end_to_end(tmp_path):
    """End-to-end GRPO smoke: 2 optimizer steps on the tiny local/base model."""
    cfg = load_config(profile="smoke")
    cfg["out_dir"] = str(tmp_path)
    cfg["rl"].update(
        {
            "max_steps": 2,
            "group_size": 2,
            "batch_size": 2,
            "train_tasks": 2,
            "init_from_sft": False,
            "output_dir_dense": str(tmp_path / "rl_dense"),
        }
    )
    cfg["sft"]["lora_r"] = 4
    from gwa.train.rl import run

    log = run(cfg, "dense")
    assert log["steps"] == 2
    assert log["trainer"] == "trl.GRPOTrainer"
    assert math.isfinite(log["final_mean_kl"])
    curve = tmp_path / "rl_dense_curve.jsonl"
    assert curve.exists()
    rows = [json.loads(line) for line in curve.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(math.isfinite(r["mean_kl"]) for r in rows)
    assert (tmp_path / "rl_dense" / "adapter_config.json").exists()
