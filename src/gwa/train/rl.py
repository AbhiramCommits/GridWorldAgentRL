"""Online RL (GRPO) over multi-turn rollouts with the environment as the reward.

Design
------
``rollout_func`` runs a *full on-policy multi-turn episode* in the real CRM
environment for every sampled generation. The environment is the reward function:
the return is computed from programmatic task success plus the configured shaping
mode (sparse terminal reward or dense per-step credit). The GRPO completion that is
optimised is the final model turn of that episode; its prompt carries the full
interaction history, so the policy is trained on the decision that closes the
episode. Logprobs are recomputed by the trainer; we therefore return ``logprobs=None``.

KL control is the GRPO ``beta`` term against the frozen reference (the adapter-
disabled base when training a PEFT adapter). Mean KL is logged every step.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

from ..agent.loop import run_episode
from ..agent.policies import HFPolicy
from ..agent.prompt import build_user_content
from ..env.reward import episode_return
from ..env.tasks import Task, load_tasks
from ..env.tools import ToolCall  # noqa: F401  (documented action type)
from ..rollout import make_env_for_task
from ..utils import REPO_ROOT, ensure_parent, mean, run_metadata, set_seed, write_json
from .common import adapter_dir, load_base_model, load_tokenizer, make_lora_config

logger = logging.getLogger(__name__)


class _TrainerPolicy(HFPolicy):
    """HFPolicy bound to the trainer's *live* model and tokenizer."""

    def __init__(self, model, tokenizer, cfg: dict[str, Any]):
        model_cfg = cfg.get("model", {})
        rl_cfg = cfg.get("rl", {})
        super().__init__(
            model_id="trainer-live",
            adapter_path=None,
            name="rl_policy",
            max_new_tokens=int(rl_cfg.get("max_new_tokens", 96)),
            temperature=float(rl_cfg.get("temperature", 0.9)),
            top_p=float(rl_cfg.get("top_p", 0.95)),
            repetition_penalty=float(model_cfg.get("repetition_penalty", 1.05)),
            device=str(model_cfg.get("device", "cpu")),
        )
        self.model = model
        self.tokenizer = tokenizer

    def _ensure_loaded(self) -> None:
        return None

    def bind_task(self, task: Task) -> None:
        return None


def _task_from_prompt(prompt: str, task_by_prompt: dict[str, Task]) -> Task:
    task = task_by_prompt.get(prompt)
    if task is not None:
        return task
    marker = "task_id="
    if marker in prompt:
        tid = prompt.split(marker, 1)[1].split()[0].strip("|, ")
        for candidate in task_by_prompt.values():
            if candidate.task_id == tid:
                return candidate
    raise KeyError("could not resolve the task for a GRPO prompt")


def _render_prefix(tokenizer, user_content: str) -> str:
    from ..train.common import render_prefix

    return render_prefix(tokenizer, user_content)


def make_rollout_func(cfg: dict[str, Any], mode: str, task_by_prompt: dict[str, Task]):
    """Return a TRL ``rollout_func`` that executes real multi-turn episodes."""

    counter = {"n": 0}

    def rollout_func(prompts: list[str], trainer) -> dict[str, Any]:
        tokenizer = trainer.processing_class
        model = trainer.accelerator.unwrap_model(trainer.model)
        was_training = model.training
        model.eval()
        policy = _TrainerPolicy(model, tokenizer, cfg)

        prompt_ids_list: list[list[int]] = []
        completion_ids_list: list[list[int]] = []
        mask_list: list[list[int]] = []
        rewards: list[float] = []
        successes: list[float] = []
        turn_counts: list[float] = []
        tool_call_counts: list[float] = []
        term_rewards: list[float] = []

        for prompt in prompts:
            counter["n"] += 1
            task = _task_from_prompt(prompt, task_by_prompt)
            env = make_env_for_task(task, cfg, reward_mode=mode)
            episode_seed = int(cfg.get("seed", 0)) + counter["n"] * 13 + int(trainer.state.global_step)
            traj = run_episode(env, task, policy, cfg, episode_seed=episode_seed)
            ret = episode_return(mode, traj.rewards, traj.terminal_reward)

            # Optimise the final model turn: its prompt contains the full history.
            turn = traj.turns[-1] if traj.turns else {
                "prompt": prompt,
                "response": "FINISH: no output",
            }
            prefix = _render_prefix(tokenizer, turn["prompt"])
            pids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            response = turn.get("response") or "FINISH: no output"
            cids = tokenizer(response, add_special_tokens=False)["input_ids"]
            cids = cids + [tokenizer.eos_token_id]
            if not pids:
                pids = [tokenizer.bos_token_id or tokenizer.eos_token_id]
            prompt_ids_list.append(pids)
            completion_ids_list.append(cids)
            mask_list.append([1] * len(cids))
            rewards.append(float(ret))
            successes.append(1.0 if traj.success else 0.0)
            turn_counts.append(float(len(traj.turns)))
            tool_call_counts.append(float(traj.tool_calls))
            term_rewards.append(float(traj.terminal_reward))

        model.train(was_training)
        return {
            "prompt_ids": prompt_ids_list,
            "completion_ids": completion_ids_list,
            "logprobs": None,
            "env_mask": mask_list,
            "episode_reward": rewards,
            "success": successes,
            "turns": turn_counts,
            "tool_calls": tool_call_counts,
            "terminal_reward": term_rewards,
        }

    return rollout_func


def make_env_reward_fn(mode: str):
    """Environment-as-reward function wired into GRPO.

    The actual reward is produced by :func:`run_episode` inside the rollout
    function: every sampled generation runs a full episode against the real CRM,
    and ``episode_return`` combines the programmatic terminal reward with the
    configured shaping mode. This function returns those cached environment
    returns and logs per-step diagnostics.
    """

    def env_reward(
        prompts: list[str],
        completions: list[str],
        episode_reward: list[float] | None = None,
        success: list[float] | None = None,
        turns: list[float] | None = None,
        terminal_reward: list[float] | None = None,
        log_metric=None,
        **kwargs,
    ) -> list[float]:
        if episode_reward is None:
            raise RuntimeError("episode_reward missing: rollout_func must run the environment")
        if log_metric is not None:
            log_metric(f"env/{mode}_success_rate", mean(success or [0.0]))
            log_metric(f"env/{mode}_mean_turns", mean(turns or [0.0]))
            log_metric(f"env/{mode}_mean_terminal_reward", mean(terminal_reward or [0.0]))
        return [float(x) for x in episode_reward]

    env_reward.__name__ = f"env_reward_{mode}"
    return env_reward


def _load_rl_dataset(cfg: dict[str, Any]):
    from datasets import Dataset

    data_cfg = cfg.get("data", {})
    train_path = REPO_ROOT / data_cfg.get("tasks_train", "data/tasks_train.jsonl")
    tasks = load_tasks(train_path)
    limit = int(cfg.get("rl", {}).get("train_tasks", 8))
    tasks = tasks[:limit]
    rows = []
    task_by_prompt: dict[str, Task] = {}
    for task in tasks:
        env = make_env_for_task(task, cfg)
        obs = env.reset()
        prompt = build_user_content(task.instruction, obs, [], cfg)
        task_by_prompt[prompt] = task
        rows.append({"prompt": prompt, "task_id": task.task_id})
    return Dataset.from_list(rows), task_by_prompt


def run(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    from trl import GRPOConfig, GRPOTrainer

    if mode not in ("sparse", "dense"):
        raise ValueError("mode must be 'sparse' or 'dense'")
    set_seed(int(cfg.get("seed", 0)))
    rl_cfg = cfg.get("rl", {})
    trainable_from_sft = bool(rl_cfg.get("init_from_sft", True))

    tokenizer = load_tokenizer(cfg)
    base = load_base_model(cfg, tokenizer)
    peft_config = None
    init_path = None
    if trainable_from_sft:
        sft_dir = adapter_dir(cfg, "sft")
        if sft_dir.exists():
            from peft import PeftModel

            base = PeftModel.from_pretrained(base, str(sft_dir))
            init_path = str(sft_dir)
        else:
            logger.warning("SFT adapter not found; RL starts from the base model")
    if init_path is None:
        peft_config = make_lora_config(cfg, "sft")

    dataset, task_by_prompt = _load_rl_dataset(cfg)
    out_dir = adapter_dir(cfg, f"rl_{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    args = GRPOConfig(
        output_dir=str(out_dir),
        max_steps=int(rl_cfg.get("max_steps", 6)),
        per_device_train_batch_size=int(rl_cfg.get("batch_size", 2)),
        gradient_accumulation_steps=int(rl_cfg.get("grad_accum", 1)),
        learning_rate=float(rl_cfg.get("lr", 5e-5)),
        num_generations=int(rl_cfg.get("group_size", 2)),
        max_completion_length=512,
        beta=float(rl_cfg.get("beta", 0.04)),
        temperature=float(rl_cfg.get("temperature", 0.9)),
        top_p=float(rl_cfg.get("top_p", 0.95)),
        logging_steps=int(rl_cfg.get("logging_steps", 1)),
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
        remove_unused_columns=False,
        scale_rewards="group",
        loss_type="grpo",
        disable_tqdm=True,
    )
    trainer = GRPOTrainer(
        model=base,
        reward_funcs=[make_env_reward_fn(mode)],
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=make_rollout_func(cfg, mode, task_by_prompt),
    )
    started = time.time()
    result = trainer.train()
    wall = time.time() - started
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    curve_path = ensure_parent(f"{cfg.get('out_dir', 'results')}/rl_{mode}_curve.jsonl")
    curve: list[dict[str, Any]] = []
    for entry in trainer.state.log_history:
        if "reward" not in entry and "kl" not in entry:
            continue
        curve.append(
            {
                "step": int(entry.get("step", len(curve) + 1)),
                "mean_reward": float(entry.get("reward", 0.0) or 0.0),
                "mean_kl": float(entry.get("kl", 0.0) or 0.0),
                "success_rate": float(entry.get(f"env/{mode}_success_rate", 0.0) or 0.0),
                "mean_turns": float(entry.get(f"env/{mode}_mean_turns", 0.0) or 0.0),
                "loss": float(entry.get("loss", 0.0) or 0.0),
            }
        )
    if not curve:
        curve = [
            {
                "step": 0,
                "mean_reward": float(getattr(result, "training_loss", 0.0) or 0.0),
                "mean_kl": 0.0,
                "success_rate": 0.0,
                "mean_turns": 0.0,
                "loss": 0.0,
            }
        ]
    with open(curve_path, "w", encoding="utf-8") as fh:
        for row in curve:
            fh.write(json.dumps(row) + "\n")

    kls = [r["mean_kl"] for r in curve]
    if any(not math.isfinite(k) for k in kls):
        raise RuntimeError(f"KL diverged in RL/{mode}: {kls}")
    rewards = [r["mean_reward"] for r in curve]
    success_rates = [r["success_rate"] for r in curve]
    reward_collapse = bool(rewards) and abs(rewards[-1]) < 1e-9 and all(
        s == 0.0 for s in success_rates
    )
    if reward_collapse:
        logger.warning(
            "reward collapsed to zero for RL/%s: the initialized policy produced no valid "
            "actions, so every group had zero advantage (tiny-model limitation)",
            mode,
        )

    log = {
        "mode": mode,
        "trainer": "trl.GRPOTrainer",
        "final_loss": float(getattr(result, "training_loss", 0.0) or 0.0),
        "steps": int(trainer.state.global_step),
        "wall_seconds": round(wall, 2),
        "n_train_prompts": len(dataset),
        "group_size": args.num_generations,
        "beta": args.beta,
        "temperature": args.temperature,
        "init_adapter": init_path,
        "curve_path": str(curve_path),
        "final_mean_reward": rewards[-1] if rewards else 0.0,
        "final_mean_kl": kls[-1] if kls else 0.0,
        "reward_collapsed": reward_collapse,
        "output_dir": str(out_dir),
        "config": {
            "max_steps": args.max_steps,
            "batch_size": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "lr": args.learning_rate,
            "group_size": args.num_generations,
            "beta": args.beta,
            "loss_type": args.loss_type,
        },
        "metadata": run_metadata(cfg),
    }
    write_json(out_dir / "train_log.json", log)
    write_json(f"{cfg.get('out_dir', 'results')}/rl_{mode}_train.json", log)
    return log
