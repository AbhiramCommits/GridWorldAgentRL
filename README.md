# GridWorldAgentRL

GridWorldAgentRL is an online reinforcement-learning post-training pipeline for a multi-turn,
tool-using LLM agent in a deterministic simulated enterprise CRM / support-ticketing environment.
The agent emits an explicit plan, calls one validated tool per turn, observes the result, and
replans when a tool fails. We compare a base instruct model, SFT, DPO, and online GRPO (with
environment-as-reward and KL control) on a held-out task split, reporting task success, planning
efficiency, recovery, and invalid-action rates with bootstrap confidence intervals, alongside a
reward-shaping ablation (sparse vs. dense) and an explicit reward-hacking audit.

Full documentation, architecture, reproduction commands, and measured results are added in Step 7.
