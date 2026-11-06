"""Policy backends: scripted oracle, Hugging Face, and optional vLLM.

vLLM is never a hard dependency. ``VLLMPolicy`` imports it lazily and falls back to
``HFPolicy`` with a warning when the extra is not installed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from ..env.tasks import Task
from ..env.tools import ToolCall
from ..utils import REPO_ROOT
from .prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@runtime_checkable
class Policy(Protocol):
    name: str

    def act(self, prompt: str) -> str:
        """Return one raw model turn for the given user-content prompt."""

    def bind_task(self, task: Task) -> None:
        """Optional hook so an oracle policy can inspect the task specification."""


# ---------------------------------------------------------------------------
# Scripted oracle / ceiling
# ---------------------------------------------------------------------------
class ScriptedPolicy:
    """Hand-coded near-optimal solver used as an oracle ceiling and fast test policy.

    It ignores the LLM prompt but respects the same action interface, so it exercises
    the real environment, tools, parser, and metric pipeline.
    """

    name = "scripted"

    def __init__(self) -> None:
        self.task: Task | None = None
        self.actions: list[ToolCall] = []
        self.plans: list[str] = []
        self.index = 0
        self.needs_replan = False

    def bind_task(self, task: Task) -> None:
        self.task = task
        self.index = 0
        self.needs_replan = False
        self.actions, self.plans = self._build(task)

    def observe_tool_result(self, ok: bool, error: str | None) -> None:
        if not ok:
            self.needs_replan = True

    def _build(self, task: Task) -> tuple[list[ToolCall], list[str]]:
        spec = task.spec
        actions: list[ToolCall] = []
        plans: list[str] = []
        if task.family == "triage_and_route":
            for target in spec["targets"]:
                actions.append(
                    ToolCall(
                        "update_ticket",
                        {
                            "ticket_id": target["ticket_id"],
                            "priority": target["expected_priority"],
                        },
                    )
                )
                plans.append(
                    f"Set {target['ticket_id']} to priority {target['expected_priority']}."
                )
        elif task.family == "resolve_with_kb":
            for target in spec["targets"]:
                actions.append(
                    ToolCall(
                        "update_ticket",
                        {
                            "ticket_id": target["ticket_id"],
                            "status": "resolved",
                            "remedy_code": target["remedy_code"],
                        },
                    )
                )
                plans.append(
                    f"Resolve {target['ticket_id']} with remedy {target['remedy_code']}."
                )
        elif task.family == "escalate_sla_breach":
            for tid in spec["targets"]:
                actions.append(
                    ToolCall("escalate_ticket", {"ticket_id": tid, "reason": "SLA breach"})
                )
                plans.append(f"Escalate {tid} for its breached SLA.")
        elif task.family == "dedupe_duplicate_tickets":
            actions.append(
                ToolCall(
                    "update_ticket",
                    {
                        "ticket_id": spec["duplicate"],
                        "status": "resolved",
                        "remedy_code": spec.get("remedy_code"),
                    },
                )
            )
            plans.append(f"Resolve duplicate {spec['duplicate']} and leave {spec['primary']} open.")
        elif task.family == "bulk_status_sweep":
            for tid in spec["targets"]:
                actions.append(ToolCall("update_ticket", {"ticket_id": tid, "status": "pending"}))
                plans.append(f"Move {tid} to pending.")
        elif task.family == "refuse_impossible_request":
            actions = []
            plans = ["Refuse the out-of-policy request without mutating state."]
        return actions, plans

    def act(self, prompt: str) -> str:
        if self.task is None:
            return "FINISH: No task was bound to the scripted policy."
        if self.task.family == "refuse_impossible_request":
            return (
                "FINISH: I cannot issue a $500 refund above the $50 policy limit or delete "
                "CRM records, so I am declining this request."
            )
        if self.index >= len(self.actions):
            return "FINISH: All requested ticket updates are complete."

        action = self.actions[self.index]
        steps = self.plans[self.index :] or ["Complete the task."]
        plan_lines = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps[:4]))
        if self.needs_replan:
            plan_lines = "1. Recover from the failed call with a corrected action.\n" + plan_lines
            self.needs_replan = False
        self.index += 1
        payload = json.dumps(action.to_dict(), ensure_ascii=True)
        return f"PLAN:\n{plan_lines}\nACTION: {payload}"


# ---------------------------------------------------------------------------
# Hugging Face backend
# ---------------------------------------------------------------------------
class HFPolicy:
    """Transformers generation policy with an optional PEFT LoRA adapter."""

    def __init__(
        self,
        model_id: str,
        adapter_path: str | Path | None = None,
        *,
        name: str = "hf",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        device: str = "cpu",
        fallback_id: str = "sshleifer/tiny-gpt2",
        local_tiny_if_unavailable: bool = True,
        local_tiny_dir: str = "assets/tiny_model",
    ):
        self.name = name
        self.model_id = model_id
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.device = device
        self.fallback_id = fallback_id
        self.local_tiny_if_unavailable = local_tiny_if_unavailable
        self.local_tiny_dir = Path(local_tiny_dir)
        if not self.local_tiny_dir.is_absolute():
            self.local_tiny_dir = REPO_ROOT / self.local_tiny_dir
        self.tokenizer = None
        self.model = None
        self.loaded_from: str | None = None

    # -- loading -------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        errors: list[str] = []
        for candidate in (self.model_id, self.fallback_id):
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                model = AutoModelForCausalLM.from_pretrained(
                    candidate, dtype=torch.float32
                )
                self.tokenizer, self.model, self.loaded_from = tokenizer, model, candidate
                break
            except Exception as exc:  # network or architecture failure
                errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
                logger.warning("could not load %s (%s)", candidate, exc)
        if self.model is None:
            if not self.local_tiny_if_unavailable:
                raise RuntimeError("no model available: " + " | ".join(errors))
            logger.warning("building a local randomly-initialized tiny GPT-2 model")
            self.tokenizer, self.model = _build_local_tiny_model(self.local_tiny_dir)
            self.loaded_from = str(self.local_tiny_dir)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.to(self.device)
        self.model.eval()

        if self.adapter_path:
            adapter = Path(self.adapter_path)
            if not adapter.is_absolute():
                adapter = REPO_ROOT / adapter
            if adapter.exists():
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, str(adapter))
                self.model.eval()
                self.loaded_from = f"{self.loaded_from}+{adapter}"
            else:
                logger.warning(
                    "adapter %s not found for policy %s; evaluating the base model instead",
                    adapter,
                    self.name,
                )

    # -- generation ----------------------------------------------------
    def _render(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{SYSTEM_PROMPT}\n\nUSER:\n{prompt}\n\nASSISTANT:\n"

    def act(self, prompt: str) -> str:
        try:
            self._ensure_loaded()
            text = self._render(prompt)
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            do_sample = self.temperature > 0
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": self.repetition_penalty,
                "stopping_criteria": StoppingCriteriaList(
                    [_StopCriteria(self.tokenizer, inputs["input_ids"].shape[1])]
                ),
            }
            if do_sample:
                gen_kwargs.update(temperature=self.temperature, top_p=self.top_p)
            with torch.inference_mode():
                output = self.model.generate(**inputs, **gen_kwargs)
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as exc:  # a policy failure must not crash a whole benchmark
            logger.warning("generation failed: %s", exc)
            return ""

    def bind_task(self, task: Task) -> None:
        return None


def _build_local_tiny_model(directory: Path):
    """Build a tiny GPT-2 + a local WordLevel tokenizer with no network access."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    directory.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = directory.parent / "tiny_tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    corpus = [
        "PLAN: ACTION: FINISH: search_tickets get_ticket search_kb update_ticket "
        "escalate_ticket reply_customer ticket customer status priority resolved open "
        "pending escalated billing technical account shipping refund remedy_code "
        "SLA breach duplicate summary policy cannot unable",
        "1. Look up the ticket. 2. Update its priority. 3. Resolve with the KB remedy.",
        "The quick brown fox jumps over the lazy dog. Enterprise support CRM sandbox.",
    ]
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(corpus, vocab_size=320)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<eos>",
    )
    fast.save_pretrained(tokenizer_dir)

    config = GPT2Config(
        n_layer=4,
        n_head=4,
        n_embd=128,
        n_positions=1024,
        vocab_size=len(fast),
        bos_token_id=fast.bos_token_id,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(directory)
    fast.save_pretrained(directory)
    logger.info("saved local tiny model to %s and tokenizer to %s", directory, tokenizer_dir)
    return fast, model


class _StopCriteria(StoppingCriteria):
    """Stop after a complete ACTION JSON or a FINISH line, to bound CPU generation."""

    def __init__(self, tokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.seen_action = False
        self.seen_finish = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Decode only a small tail: full-sequence decoding at every step dominates CPU time.
        tail = self.tokenizer.decode(input_ids[0, -12:], skip_special_tokens=True)
        if "ACTION:" in tail:
            self.seen_action = True
        if "FINISH:" in tail:
            self.seen_finish = True
        if self.seen_action and "}" in tail:
            return True
        if self.seen_finish and "\n" in tail:
            return True
        return False


def _balanced(text: str) -> bool:
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


# ---------------------------------------------------------------------------
# Optional vLLM backend
# ---------------------------------------------------------------------------
class VLLMPolicy:
    """vLLM generation when available, otherwise a warned fallback to HFPolicy."""

    def __init__(self, model_id: str, adapter_path: str | Path | None = None, **kwargs: Any):
        self.name = kwargs.pop("name", "vllm")
        self.fallback: HFPolicy | None = None
        try:
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            from vllm import LLM, SamplingParams  # noqa: F401

            self._LLM = LLM
            self._SamplingParams = SamplingParams
            lora_kwargs = {}
            if adapter_path and Path(adapter_path).exists():
                lora_kwargs = {"enable_lora": True, "max_lora_rank": 16}
            self.llm = LLM(model=model_id, dtype="float32", **lora_kwargs)
            self.tokenizer = self.llm.get_tokenizer()
            self.adapter_path = str(adapter_path) if adapter_path else None
            self.max_new_tokens = kwargs.get("max_new_tokens", 64)
            self.temperature = kwargs.get("temperature", 0.0)
            self.top_p = kwargs.get("top_p", 0.95)
        except Exception as exc:
            logger.warning(
                "vLLM requested but unavailable (%s); falling back to HFPolicy", exc
            )
            self.fallback = HFPolicy(model_id, adapter_path, **kwargs)

    def bind_task(self, task: Task) -> None:
        if self.fallback:
            self.fallback.bind_task(task)

    def act(self, prompt: str) -> str:
        if self.fallback is not None:
            return self.fallback.act(prompt)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        params = self._SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["\nACTION:", "\nFINISH:"],
        )
        outputs = self.llm.generate([text], params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def policy_adapter_path(name: str, cfg: dict[str, Any]) -> str | None:
    mapping = {
        "sft": cfg.get("sft", {}).get("output_dir", "checkpoints/sft"),
        "dpo": cfg.get("dpo", {}).get("output_dir", "checkpoints/dpo"),
        "rl_sparse": cfg.get("rl", {}).get("output_dir_sparse", "checkpoints/rl_sparse"),
        "rl_dense": cfg.get("rl", {}).get("output_dir_dense", "checkpoints/rl_dense"),
    }
    return mapping.get(name)


def build_policy(
    name: str,
    cfg: dict[str, Any],
    backend: str = "hf",
    task: Task | None = None,
    adapter_path: str | Path | None = None,
) -> Policy:
    """Construct a policy by name, resolving adapters from the config."""
    model_cfg = cfg.get("model", {})
    if name == "scripted":
        policy = ScriptedPolicy()
    else:
        adapter = adapter_path or policy_adapter_path(name, cfg)
        kwargs = dict(
            name=name,
            max_new_tokens=int(model_cfg.get("max_new_tokens", 64)),
            temperature=float(model_cfg.get("temperature", 0.0)),
            top_p=float(model_cfg.get("top_p", 0.95)),
            repetition_penalty=float(model_cfg.get("repetition_penalty", 1.05)),
            device=str(model_cfg.get("device", "cpu")),
            fallback_id=str(model_cfg.get("fallback_id", "sshleifer/tiny-gpt2")),
            local_tiny_if_unavailable=bool(model_cfg.get("local_tiny_if_unavailable", True)),
            local_tiny_dir=str(model_cfg.get("local_tiny_dir", "assets/tiny_model")),
        )
        model_id = str(model_cfg.get("id", "HuggingFaceTB/SmolLM2-135M-Instruct"))
        if backend == "vllm":
            policy = VLLMPolicy(model_id, adapter, **kwargs)
        else:
            if backend not in ("hf", "base"):
                logger.warning("unknown backend %r; using hf", backend)
            policy = HFPolicy(model_id, adapter, **kwargs)
    if task is not None:
        policy.bind_task(task)
    return policy
