"""LoRA DPO with TRL DPOTrainer (reference = adapter-disabled base policy)."""

from __future__ import annotations

import time
from typing import Any

from ..data.build_prefs import load_pairs
from ..utils import run_metadata, set_seed, write_json
from .common import (
    adapter_dir,
    count_trainable_params,
    load_base_model,
    load_tokenizer,
    make_lora_config,
)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    set_seed(int(cfg.get("seed", 0)))
    sec = cfg.get("dpo", {})
    pairs = load_pairs(cfg)
    if not pairs:
        raise RuntimeError("data/prefs.jsonl is empty; run `gwa build-prefs` first")

    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = adapter_dir(cfg, "dpo")
    out_dir.mkdir(parents=True, exist_ok=True)

    args = DPOConfig(
        output_dir=str(out_dir),
        max_steps=int(sec.get("max_steps", 10)),
        per_device_train_batch_size=int(sec.get("batch_size", 2)),
        gradient_accumulation_steps=int(sec.get("grad_accum", 2)),
        learning_rate=float(sec.get("lr", 5e-5)),
        beta=float(sec.get("beta", 0.1)),
        max_length=int(sec.get("max_length", 1024)),
        precompute_ref_log_probs=bool(sec.get("precompute_ref_log_probs", True)),
        logging_steps=int(sec.get("logging_steps", 2)),
        save_strategy="no",
        report_to="none",
        seed=int(cfg.get("seed", 0)),
        data_seed=int(cfg.get("seed", 0)),
        use_cpu=str(cfg.get("model", {}).get("device", "cpu")) == "cpu",
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        disable_tqdm=True,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=Dataset.from_list(pairs),
        processing_class=tokenizer,
        peft_config=make_lora_config(cfg, "dpo"),
    )
    started = time.time()
    result = trainer.train()
    wall = time.time() - started
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    trainable, total = count_trainable_params(trainer.model)
    log = {
        "final_loss": float(getattr(result, "training_loss", 0.0) or 0.0),
        "steps": int(trainer.state.global_step),
        "wall_seconds": round(wall, 2),
        "n_pairs": len(pairs),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_fraction": round(trainable / max(1, total), 6),
        "loss_history": [h["loss"] for h in trainer.state.log_history if "loss" in h],
        "output_dir": str(out_dir),
        "config": {
            "max_steps": args.max_steps,
            "batch_size": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "lr": args.learning_rate,
            "beta": args.beta,
            "lora_r": sec.get("lora_r", 8),
            "max_length": args.max_length,
            "trainer": "trl.DPOTrainer",
            "reference": "adapter-disabled base (PEFT)",
        },
        "metadata": run_metadata(cfg),
    }
    write_json(out_dir / "train_log.json", log)
    write_json(f"{cfg.get('out_dir', 'results')}/train_dpo.json", log)
    return log
