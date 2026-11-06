"""Shared model/tokenizer loading and LoRA configuration for training and data builds."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ..agent.policies import _build_local_tiny_model
from ..agent.prompt import SYSTEM_PROMPT
from ..utils import REPO_ROOT

logger = logging.getLogger(__name__)


def _candidate_ids(cfg: dict[str, Any]) -> list[str]:
    model_cfg = cfg.get("model", {})
    ids = [str(model_cfg.get("id", "HuggingFaceTB/SmolLM2-135M-Instruct"))]
    fallback = model_cfg.get("fallback_id")
    if fallback:
        ids.append(str(fallback))
    return ids


def load_tokenizer(cfg: dict[str, Any]):
    from transformers import AutoTokenizer

    errors = []
    for candidate in _candidate_ids(cfg):
        try:
            tok = AutoTokenizer.from_pretrained(candidate)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok
        except Exception as exc:
            errors.append(str(exc))
            logger.warning("tokenizer load failed for %s: %s", candidate, exc)
    local_tiny = REPO_ROOT / str(cfg.get("model", {}).get("local_tiny_dir", "assets/tiny_model"))
    tokenizer_dir = local_tiny.parent / "tiny_tokenizer"
    if not tokenizer_dir.exists() and cfg.get("model", {}).get("local_tiny_if_unavailable", True):
        logger.warning("building local tiny model and tokenizer")
        _, tok = _build_local_tiny_model(local_tiny)
        return tok
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(cfg: dict[str, Any], tokenizer):
    from transformers import AutoModelForCausalLM

    dtype = torch.float32
    errors = []
    for candidate in _candidate_ids(cfg):
        try:
            model = AutoModelForCausalLM.from_pretrained(candidate, dtype=dtype)
            logger.info("loaded base model %s", candidate)
            return model
        except Exception as exc:
            errors.append(str(exc))
            logger.warning("model load failed for %s: %s", candidate, exc)
    local_tiny = REPO_ROOT / str(cfg.get("model", {}).get("local_tiny_dir", "assets/tiny_model"))
    if cfg.get("model", {}).get("local_tiny_if_unavailable", True):
        logger.warning("using local randomly-initialized tiny GPT-2 (%s)", " | ".join(errors))
        tok, model = _build_local_tiny_model(local_tiny)
        return model
    raise RuntimeError("no model could be loaded")


def render_prefix(tokenizer, user_content: str) -> str:
    """Exact prompt prefix shared by SFT/DPO data and HFPolicy inference."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\nUSER:\n{user_content}\n\nASSISTANT:\n"


def make_lora_config(cfg: dict[str, Any], section: str):
    from peft import LoraConfig

    sec = cfg.get(section, {})
    return LoraConfig(
        r=int(sec.get("lora_r", 8)),
        lora_alpha=int(sec.get("lora_alpha", 16)),
        lora_dropout=float(sec.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )


def count_trainable_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def adapter_dir(cfg: dict[str, Any], section: str) -> Path:
    path = Path(cfg.get(section, {}).get("output_dir", f"checkpoints/{section}"))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path
