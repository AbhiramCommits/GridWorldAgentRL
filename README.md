# GridWorldAgentRL

**Online RL post-training for a multi-turn, tool-using LLM agent in a verifiable simulated enterprise environment.**

GridWorldAgentRL trains and compares four policies — a base instruct model, LoRA SFT, LoRA DPO,
and online GRPO with the environment as the reward — on a deterministic mock CRM / support-ticketing
sandbox. The agent emits an explicit numbered plan, calls exactly one validated tool per turn,
observes the result, and replans on failure. Task success is checked programmatically over the final
CRM state, never by a human or a learned reward model, and planning/replanning is measured separately
from success (`plan_step_efficiency`, `recovery_rate`, `replan_rate`). The repo includes a
reward-shaping ablation (sparse terminal reward vs. dense step reward), bootstrap confidence intervals,
and an explicit reward-hacking audit.

Everything here runs on CPU/Apple Silicon and the numbers in [RESULTS](#results) were measured by this
build. The honest headline is a **negative result at this scale**: the scripted oracle solves the
environment (1.000), SFT reaches 0.333 on the held-out split, and 6 steps of GRPO do not move the
policy beyond its SFT initialization. The environment and harness work; the 135M model does not.

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
                         │            CRMEnv (mock CRM)             │
                         │  6 tools · 6 task families · programmatic subgoals  │
                         │  deterministic state · programmatic check │
                         └───────────────┬──────────────────────────┘
                                         │ observation / reward / info
                         ┌───────────────▼──────────────────────────┐
                         │        run_episode (agent loop)          │
                         │  PLAN → ACTION → OBSERVE → REPLAN        │
                         │  records plans, calls, failures, replans │
                         └───────────────┬──────────────────────────┘
                                         │ prompt string / raw response
              ┌──────────────────────────┼───────────────────────────┐
              │                          │                           │
      ScriptedPolicy              HFPolicy (LoRA)             VLLMPolicy (optional)
      oracle / ceiling            SmolLM2-135M-Instruct       lazy import; falls
      hand-coded solver           transformers generate       back to HFPolicy
```

Training paths:

```
base SmolLM2-135M-Instruct
  ├── SFT  (LoRA r=8, 40 steps, completion-only) ──> checkpoints/sft  ──> eval "sft"
  ├── DPO  (LoRA r=8, 8 steps, beta=0.1, ref=base) ──> checkpoints/dpo ──> eval "dpo"
  └── (from SFT adapter) GRPO, env-as-reward, beta=0.04, 6 steps
          ├── sparse terminal reward ──> checkpoints/rl_sparse ──> eval "rl_sparse"
          └── dense step reward      ──> checkpoints/rl_dense  ──> eval "rl_dense"
```

Modules under `src/gwa/`:

| Module | What it does |
|---|---|
| `env/state.py` | `Customer`, `Ticket`, `KBArticle`, `CRMState`; seeded `make_state()` with a fixed solvable skeleton (breached SLAs, a duplicate pair, a resolved ticket). |
| `env/tools.py` | The six tools with strict validation and realistic failures. Mutating calls are logged for the audit. |
| `env/crm_env.py` | Gymnasium-style `reset`/`step`; `render_text()` is the compact text observation; info carries `tool_ok`, `error`, `progress`, `invalid_action`. |
| `env/tasks.py` | Six task families, programmatic `success_check`, disjoint `generate_split`, CLI `--dump`. |
| `env/reward.py` | Sparse terminal reward and dense per-step reward behind one `RewardConfig`. |
| `agent/prompt.py` | System prompt, strict PLAN/ACTION/FINISH format, token-budgeted history window. |
| `agent/parser.py` | Tolerant PLAN/ACTION/FINISH parser (markdown fences, single quotes, trailing commas, nested braces). |
| `agent/policies.py` | `ScriptedPolicy`, `HFPolicy` (LoRA adapter path), `VLLMPolicy` (optional, lazy). |
| `agent/loop.py` | `run_episode` → `Trajectory`; detects a replan when the plan changes after a failure. |
| `metrics.py` | Task success, plan-step efficiency, recovery, invalid action, malformed output, turns, reward. |
| `rollout.py` | `collect()` writes `results/rollouts_<tag>.jsonl` and a metrics summary. |
| `data/build_sft.py` | Scripted TRAIN trajectories → chat SFT data; recovery samples. |
| `data/build_prefs.py` | Chosen = scripted turn; rejected = base-policy malformed/invalid/lower-progress turn; header records the rule. |
| `train/sft.py`, `train/dpo.py` | TRL `SFTTrainer` / `DPOTrainer` with PEFT LoRA; saves adapters + `train_log.json`. |
| `train/rl.py` | TRL `GRPOTrainer` with a `rollout_func` that runs full multi-turn episodes; `make_env_reward_fn(mode)`; KL control. |
| `eval.py`, `stats.py` | Held-out benchmark, percentile bootstrap CIs, paired bootstrap deltas. |
| `audit.py` | Six named reward-hacking detectors, per-policy incidence, stacked figure, positive controls. |
| `report.py` | Assembles `results/RESULTS.md` from result files only. |
| `cli.py` | `rollout`, `build-sft`, `build-prefs`, `train-sft`, `train-dpo`, `train-rl`, `eval`, `report`, `build-tiny`. |

---

## Environment and reward design

The sandbox contains 8 customers, 20 tickets, and 10 KB articles. Tickets have `status` (open/pending/
escalated/resolved), `priority`, `category`, `sla_hours_remaining`, and text. Every task is built from
a deterministic `(seed, family)` pair; the same seed always produces the same state.

**Six tools** (all arguments validated; failures are realistic):

| Tool | Required | Optional / behavior |
|---|---|---|
| `search_tickets` | — | `query`, `customer_id`, `status` |
| `get_ticket` | `ticket_id` | full ticket + customer |
| `search_kb` | `query` | matches category, remedy code, title/body |
| `update_ticket` | `ticket_id` | `status`, `priority`, `remedy_code`; rejects resolved tickets, wrong-category remedies, and no-ops |
| `escalate_ticket` | `ticket_id`, `reason` | rejects resolved/already-escalated |
| `reply_customer` | `ticket_id`, `message` | appends a reply; does not resolve |

**Six task families** and their programmatic checks:

| Family | Goal | Success check |
|---|---|---|
| `triage_and_route` | set the right priority from SLA | target ticket has the expected priority |
| `resolve_with_kb` | resolve with the category's exact remedy | status `resolved` **and** matching `remedy_code` |
| `escalate_sla_breach` | escalate breached tickets only | targets escalated with a reason, no healthy ticket escalated |
| `dedupe_duplicate_tickets` | resolve the duplicate, keep the primary | duplicate resolved, primary not resolved |
| `bulk_status_sweep` | move one customer's open tickets to pending | all targets pending, no collateral changes |
| `refuse_impossible_request` | refuse an out-of-policy request | no mutating call **and** FINISH explains the refusal |

Splits are disjoint on `(family, seed)` and `generate_split` asserts this on every call. `eval` never
reads a train seed.

**Rewards.**

- Sparse terminal: `1.0` on programmatic success, else `0.0`, minus `0.10 × invalid tool calls`.
- Dense per step: `0.25 × Δ(subgoal progress) + 0.02 × [valid tool] − 0.01`; episode return is the
  sum of step rewards plus the terminal reward. A malformed model turn does not advance the
  environment, so it earns no step credit.

The dense return is intentionally additive with the terminal term; the ablation changes *only* the
step shaping, not the success criterion.

---

## How to run

```bash
# Python 3.11 (uv recommended)
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
# or: python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"

make tasks          # data/tasks_train.jsonl + data/tasks_eval.jsonl
make sft-data       # data/sft.jsonl
make sft            # checkpoints/sft (+ results/train_sft.json)
make prefs          # data/prefs.jsonl
make dpo            # checkpoints/dpo (+ results/train_dpo.json)
make rl-sparse      # checkpoints/rl_sparse + results/rl_sparse_curve.jsonl
make rl-dense       # checkpoints/rl_dense  + results/rl_dense_curve.jsonl
make eval           # results/eval_episodes.jsonl, eval_table.{json,md}
make report         # results/RESULTS.md from files
# or the whole pipeline:
make all
```

The profile switch is `PROFILE=smoke|default|large` (default `smoke`). The smoke profile is what this
README reports; `default` and `large` use more tasks/steps but are not the measured configuration.

**Measured smoke wall-clock on this laptop (Apple M3, 16 GB, CPU, 4 torch threads):** the sum of the
stage wall-times recorded in `results/` is
`586 s SFT + 165 s DPO + 318 s RL sparse + 150 s RL dense + 554 s eval + ~155 s data builds ≈ 32 min`.
It is not a 3-minute build on a 135M CPU model; the profile is "small", not instant. The scripted-only
rollout is ~1 s; the Docker scripted check is a few seconds.

**vLLM is optional.** `--backend vllm` is accepted by `rollout`, `eval`, and `train-rl`; if `vllm`
cannot be imported, `VLLMPolicy` logs a warning and falls back to `HFPolicy`. It is never a hard
dependency:

```bash
python -m gwa.cli rollout --policy base --split eval --backend vllm
make rl-dense BACKEND=vllm
```

**Docker** (CPU torch, entrypoint `python -m gwa.cli`):

```bash
docker build -t gridworldagentrl .
docker run --rm gridworldagentrl rollout --policy scripted --profile smoke
docker compose run --rm gwa report --profile smoke   # mounts ./results
```

**Tests:**

```bash
make test        # fast tests (pytest -m "not slow")
make test-all    # includes the 2-step GRPO end-to-end test
make lint
make coverage    # pytest --cov=gwa
```

---

## RESULTS

Every number below is copied from `results/eval_table.md`, `results/eval_table.json`,
`results/reward_hacking_audit.json`, `results/rl_*_curve.jsonl`, and `results/RESULTS.md`.
The generated report is reproducible with `make report`.

### Main held-out table

Model `HuggingFaceTB/SmolLM2-135M-Instruct`, profile `smoke`, eval seed `[0]`, **6 held-out tasks**
(one per family), 36 episodes. Success CIs are percentile bootstrap 95% (10,000 resamples).
`scripted` is the oracle ceiling, not a learned policy.

| Policy | Task success (95% CI) | Plan-step efficiency | Recovery rate | Invalid action rate | Mean turns |
|---|---|---|---|---|---|
| base | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 3.00 |
| sft | 0.333 [0.000, 0.667] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2.33 |
| dpo | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 3.00 |
| rl_sparse | 0.333 [0.000, 0.667] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2.33 |
| rl_dense | 0.333 [0.000, 0.667] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2.33 |
| scripted | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.17 |

The two SFT successes are `escalate_sla_breach` (one valid escalate call) and
`refuse_impossible_request` (a clean FINISH). No multi-step family succeeded. `recovery_rate` is 0.000
because no held-out episode both hit a tool failure and recovered; the denominator is zero, so the
metric is reported as 0.000 rather than omitted.

### Paired bootstrap comparisons

Paired on `(task_id, seed)`, same 6 pairs; Δ success is `a − b`. Positive favors `a`.

| Comparison | Δ success [95% CI] | P(a > b) | Δ efficiency [95% CI] |
|---|---|---|---|
| rl_sparse − base | +0.333 [+0.000, +0.667] | 0.911 | +0.333 [+0.000, +0.667] |
| rl_dense − base | +0.333 [+0.000, +0.667] | 0.911 | +0.333 [+0.000, +0.667] |
| rl_sparse − sft | +0.000 [+0.000, +0.000] | 0.000 | +0.000 [+0.000, +0.000] |
| rl_dense − sft | +0.000 [+0.000, +0.000] | 0.000 | +0.000 [+0.000, +0.000] |
| rl_sparse − dpo | +0.333 [+0.000, +0.667] | 0.911 | +0.333 [+0.000, +0.667] |
| rl_dense − dpo | +0.333 [+0.000, +0.667] | 0.911 | +0.333 [+0.000, +0.667] |
| rl_dense − rl_sparse | +0.000 [+0.000, +0.000] | 0.000 | +0.000 [+0.000, +0.000] |
| sft − base | +0.333 [+0.000, +0.667] | 0.911 | +0.333 [+0.000, +0.667] |
| dpo − base | +0.000 [+0.000, +0.000] | 0.000 | +0.000 [+0.000, +0.000] |

The confidence intervals are wide because the smoke split is one task per family. The RL-vs-SFT
comparison is exactly zero: the two RL adapters and the SFT adapter produce identical held-out
trajectories.

### Sparse-vs-dense shaping ablation

Both runs are 6 optimizer steps, group size 2, beta 0.04, from the same SFT adapter. Values are the
final rows of `results/rl_sparse_curve.jsonl` / `results/rl_dense_curve.jsonl`.

| Shaping | Final mean reward | Final success rate | Final mean KL | Steps | Reward collapsed |
|---|---|---|---|---|---|
| sparse | 0.000 | 0.000 | 0.000112 | 6 | True |
| dense | 0.000 | 0.000 | 0.000112 | 6 | True |

The first (and only non-zero) mean reward was −0.050 (sparse) / −0.055 (dense) at step 1; from step 2
on, every group had zero reward, so the group-normalized advantage was zero and the policy did not
move. KL never diverged (`max` KL over both curves is 0.000669; final 0.000112).

![RL learning curves](results/figures/rl_curves.png)

### Reward-hacking audit

Incidence is the fraction of a policy's 6 held-out episodes in which the named detector fired.
The positive controls are synthetic trajectories crafted to trigger each detector; all six fire.

| Detector | base | dpo | rl_dense | rl_sparse | scripted | sft |
|---|---|---|---|---|---|---|
| premature_finish | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| status_spam | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| escalation_abuse | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| reply_without_resolve | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| tool_thrash | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| over_action_on_refusal_tasks | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **any detector** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Positive controls: `premature_finish`=yes, `status_spam`=yes, `escalation_abuse`=yes,
`reply_without_resolve`=yes, `tool_thrash`=yes, `over_action_on_refusal_tasks`=yes.

**Zero incidence is not evidence of alignment.** Base, DPO, and both RL policies emitted no valid tool
calls at all on most episodes (malformed-output rate 1.000 for base and DPO), so the tool-level
detectors had little opportunity to fire. The audit is working; the policies are mostly not acting.

![Reward-hacking audit](results/figures/audit.png)

### Exact configuration

| Item | Value |
|---|---|
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` (134.98M params) |
| LoRA | r=8, alpha=16, dropout=0.05, target `q_proj`/`v_proj`; 460,800 trainable params (0.34%) |
| SFT | 40 steps, batch 4, lr 5e-4, completion-only loss, max length 768, 113 scripted demos |
| DPO | 8 steps, batch 1 × grad-accum 2, lr 5e-5, beta 0.1, 16 pairs, ref = adapter-disabled base |
| GRPO | 6 steps, group size 2, batch 2, lr 5e-5, beta 0.04, temperature 0.9, max 6 turns, SFT init |
| Eval | 6 tasks × 1 seed (0), temperature 0.2, bootstrap n=10,000 |
| Seeds | data/model/eval seed 0; episode seeds derived from the config seed; every result file records the seed |
| Hardware | Apple M3, 16 GB, macOS, CPU (no CUDA), 4 torch threads |
| Coverage | **54%** line coverage with `pytest -q -m "not slow" --cov=gwa` (recorded in `results/coverage.txt`; 70 fast tests, 71 total including the slow 2-step GRPO test) |
| Result commit | `f39d69f` (results); docs/tests/CI commits follow without changing measured numbers |

Training cost from the logs:

| Stage | Trainer | Steps | Wall seconds | Final loss |
|---|---|---|---|---|
| SFT | `trl.SFTTrainer` | 40 | 586.1 | 0.9284 |
| DPO | `trl.DPOTrainer` | 8 | 165.5 | 0.6605 |
| RL sparse | `trl.GRPOTrainer` | 6 | 318.2 | 0.0000145 |
| RL dense | `trl.GRPOTrainer` | 6 | 150.4 | 0.0000145 |

---

## Findings

- **The environment is solvable and its rewards are verifiable.** The scripted solver succeeds on
  all 6 held-out tasks (1.000) with plan-step efficiency 1.000, and every success is checked by
  inspecting the final `CRMState`. The harness, tools, and success checks are not the bottleneck.
- **SFT is the only learned policy with non-zero held-out success (0.333 [0.000, 0.667]), and it
  succeeds only on single-action families.** Its two wins are `escalate_sla_breach` and
  `refuse_impossible_request`; it fails every multi-step family. The 33% is a copy-the-explicit-action
  result, not evidence of planning. Plan-step efficiency is 1.000 on the few successes because the
  optimal action count is one.
- **DPO did not improve over base (0.000).** DPO ran 8 steps to a final loss of 0.6605
  (`results/train_dpo.json`), but the resulting policy still produced 3 malformed turns per held-out
  episode, identical to base. Sixteen pairs of (scripted chosen, malformed rejected) taught the
  optimizer a preference in the training distribution that did not transfer to tool selection on
  held-out states.
- **Online GRPO did not move the policy.** Sparse and dense RL are numerically identical to SFT on
  every held-out metric (`rl_* − sft = 0.000 [0.000, 0.000]`). The mean reward collapses to 0 after
  step 1, i.e. every group of 2 rollouts had zero reward and therefore zero advantage. KL stayed
  finite and small (final 1.1e-4), so this is a **signal** failure, not an optimization blow-up.
- **Dense shaping did not improve sample efficiency in this regime.** Sparse and dense have the same
  final reward (0.000) and KL (0.000112); the only difference is the sign of the first-step reward.
  Dense credit cannot help when the policy almost never produces a valid action, so the reward's
  support is never entered. This is a negative result for dense shaping at this scale.
- **No reward hacking was observed, but the audit is not exonerating the policies.** All measured
  detector rates are 0.000, while all six synthetic positive controls fire. Because base/DPO/RL
  policies rarely emitted a tool call, the absence of `status_spam`, `escalation_abuse`, etc. is
  mostly an absence of behavior. The audit would need a policy that actually acts to be informative.

### Limitations

- **Scale:** 135M parameters, LoRA on q/v only, 113 SFT demos, 16 DPO pairs, 6 GRPO steps. This is a
  pipeline demonstration, not a competitive agent.
- **Episode budget:** 6 held-out tasks (one per family), one eval seed, group size 2, 6 turns. The
  bootstrap CIs are correspondingly wide and should not be read as precise estimates.
- **Synthetic environment:** the CRM, its tools, and its success checks are hand-built. Results do not
  transfer to a real ticketing system without re-validating the tool contracts and task checks.
- **No planning result:** the only successes are single-action; the plan/replan machinery is
  exercised and measured, but no learned policy demonstrates multi-step recovery.
- **Recovery demonstrations:** `build_sft` synthesizes tool-error and malformed-output recovery
  turns, but the measured smoke run used a 768-token sequence cap and those longer recovery prompts
  were filtered out (`data/sft_meta.json` reports `n_recovery: 0` for that run). Raising
  `sft.max_length` to 832 admits 58 recovery samples in a fresh build (178 training records); that
  larger run was not the one evaluated here.
- **Coverage:** 54% line coverage on the fast suite (`results/coverage.txt`); the slow GRPO path is
  tested but not counted in that number.
- **RL implementation detail:** GRPO optimizes the final model turn of an on-policy multi-turn
  episode, with the full history in that turn's prompt. This is a valid full-episode-reward
  formulation but assigns credit coarsely across turns.

---

## Reproducibility

The measured run used seed 0 throughout, 6 held-out tasks (one per family), and the `smoke` profile.
The result files record `profile`, `seed`, `model_id`, and `git_commit`; the measured tree is at
commit `f39d69f`, with later test/CI/docs commits not changing the numbers. `make all` regenerates
every result file; `make report` rebuilds `results/RESULTS.md` from them.

Exact commands that produced the committed files:

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
make all                 # tasks -> sft-data -> sft -> prefs -> dpo -> rl-sparse -> rl-dense -> eval -> report
make test                # 70 fast tests
make test-all            # + 1 slow 2-step GRPO end-to-end test
.venv/bin/ruff check .
.venv/bin/python -m pytest -q -m "not slow" --cov=gwa --cov-report=term
docker build -t gridworldagentrl .
docker run --rm gridworldagentrl rollout --policy scripted --profile smoke
```

Artifacts:

- `results/eval_table.{json,md}` — held-out metrics, CIs, paired comparisons.
- `results/eval_episodes.jsonl` — one record per policy × task × seed.
- `results/reward_hacking_audit.{json,md}` — detector definitions, incidence, positive controls.
- `results/rl_{sparse,dense}_curve.jsonl` — step, reward, KL, success, turns.
- `results/figures/rl_curves.png`, `results/figures/audit.png`.
- `results/train_{sft,dpo}.json`, `results/rl_{sparse,dense}_train.json` — trainer, steps, loss, wall time.
- `results/RESULTS.md` — generated report with the main table, paired comparisons, ablation, audit.
- `checkpoints/{sft,dpo,rl_sparse,rl_dense}/` — LoRA adapters (not committed; regenerate with `make all`).

The code is MIT licensed. See `LICENSE`.
