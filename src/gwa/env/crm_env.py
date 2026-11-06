"""Gymnasium-style wrapper around the mock CRM.

Only the Space types come from Gymnasium; the environment itself is a plain,
fully deterministic Python class so it can run anywhere without a simulator.
"""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from .reward import RewardConfig, dense_reward, task_progress, terminal_reward
from .state import CATEGORY_REMEDY, CRMState, make_state
from .tasks import DEFAULT_ENV_SIZES, Task
from .tools import TOOL_NAMES, ToolCall, execute_tool

Observation = str


class CRMEnv:
    """Deterministic support-ticketing environment with verifiable rewards."""

    metadata = {"render_modes": ["text"]}

    def __init__(
        self,
        task: Task,
        max_steps: int = 8,
        reward_mode: str = "dense",
        reward_cfg: RewardConfig | dict[str, Any] | None = None,
        env_sizes: dict[str, int] | None = None,
    ):
        self.task = task
        self.max_steps = int(max_steps)
        self.reward_mode = reward_mode
        if isinstance(reward_cfg, dict):
            reward_cfg = RewardConfig.from_dict(reward_cfg)
        self.reward_cfg = reward_cfg or RewardConfig()
        self.env_sizes = env_sizes or task.env_sizes or dict(DEFAULT_ENV_SIZES)

        self.action_space = spaces.Dict(
            {
                "tool": spaces.Text(min_length=1, max_length=32),
                "args_json": spaces.Text(max_length=1024),
            }
        )
        self.observation_space = spaces.Text(max_length=16384)

        self.state: CRMState | None = None
        self.turn = 0
        self.done = False
        self.history: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None
        self.initial_hash: str | None = None
        self._prev_progress = 0.0

    # ------------------------------------------------------------------
    # Gymnasium-style API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Observation:
        self.state = make_state(
            self.task.seed if seed is None else int(seed),
            n_customers=int(self.env_sizes.get("n_customers", 8)),
            n_tickets=int(self.env_sizes.get("n_tickets", 20)),
            n_articles=int(self.env_sizes.get("n_articles", 10)),
        )
        self.turn = 0
        self.done = False
        self.history = []
        self.last_result = None
        self.initial_hash = self.state.state_hash()
        _, _, self._prev_progress = task_progress(self.task, self.state)
        return self.render_text()

    def step(
        self, action: ToolCall
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        if self.done:
            info = {
                "tool_ok": False,
                "error": "episode already finished",
                "progress": self._prev_progress,
                "invalid_action": True,
                "terminal": True,
            }
            return self.render_text(), 0.0, True, True, info

        if not isinstance(action, ToolCall):
            action = ToolCall.from_dict(action)  # type: ignore[arg-type]

        self.turn += 1
        self.state.current_turn = self.turn
        result = execute_tool(self.state, action)
        ok, flags, progress = task_progress(self.task, self.state)
        info: dict[str, Any] = {
            "tool_ok": bool(result.ok),
            "error": result.error,
            "progress": float(progress),
            "prev_progress": float(self._prev_progress),
            "invalid_action": not result.ok,
            "result": result.to_dict(),
            "success": bool(ok),
            "subgoals": flags,
        }
        step_reward = (
            dense_reward(self.task, self.state, info, self.turn, self.reward_cfg)
            if self.reward_mode == "dense"
            else 0.0
        )
        self._prev_progress = float(progress)
        self.last_result = info
        self.history.append(
            {
                "turn": self.turn,
                "tool": action.tool,
                "args": action.args,
                "ok": result.ok,
                "error": result.error,
                "payload": _compact_payload(result.payload),
            }
        )

        terminated = bool(ok)
        truncated = bool(self.turn >= self.max_steps and not terminated)
        self.done = terminated or truncated
        info["turn"] = self.turn
        return self.render_text(), float(step_reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_text(self) -> Observation:
        if self.state is None:
            return ""
        lines: list[str] = []
        lines.append(f"TASK: {self.task.instruction}")
        lines.append(f"STEP: {self.turn}/{self.max_steps} | task_id={self.task.task_id}")
        if self.last_result is not None:
            lr = self.last_result
            if lr.get("tool_ok"):
                lines.append(f"LAST: OK {_compact_payload(lr.get('result', {}).get('payload'))}")
            else:
                lines.append(f"LAST: ERROR {lr.get('error')}")
        elif self.history:
            lines.append("LAST: (none)")

        visible = [t for t in self.state.sorted_tickets() if t.status != "resolved"]
        hidden = len(self.state.tickets) - len(visible)
        lines.append(f"TICKETS ({len(visible)} active, {hidden} resolved hidden):")
        for t in visible[:6]:
            lines.append(
                f"{t.ticket_id}|{t.customer_id}|{t.status}|{t.priority}|{t.category}|"
                f"SLA{t.sla_hours_remaining}h|{_clip(t.text, 40)}"
            )
        if len(visible) > 6:
            lines.append(f"... {len(visible) - 6} more active tickets (use search_tickets)")

        kb = ", ".join(f"{c}->{CATEGORY_REMEDY[c]}" for c in sorted(CATEGORY_REMEDY))
        lines.append(f"KB REMEDY CODES: {kb}")
        if self.history:
            recent = self.history[-2:]
            lines.append("RECENT CALLS:")
            for h in recent:
                status = "ok" if h["ok"] else f"error={h['error']}"
                lines.append(f"  step{h['turn']} {h['tool']} -> {status}")
        if self.state.finish_summary:
            lines.append(f"FINISHED: {_clip(self.state.finish_summary, 200)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def finish(self, summary: str) -> None:
        """Record the agent's FINISH summary (used by refusal checks)."""
        if self.state is not None:
            self.state.finish_summary = summary
        self.done = True

    def final_terminal_reward(self) -> float:
        if self.state is None:
            return 0.0
        return terminal_reward(self.task, self.state, self.reward_cfg)

    def success(self) -> tuple[bool, dict[str, bool]]:
        if self.state is None:
            return False, {}
        return self.task.success_check(self.state)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return TOOL_NAMES

    def close(self) -> None:
        return None


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _compact_payload(payload: Any, max_items: int = 4) -> str:
    """Compact, token-cheap rendering of a tool payload for observations."""
    if payload is None:
        return "null"
    if isinstance(payload, dict):
        if "tickets" in payload and isinstance(payload["tickets"], list):
            items = payload["tickets"][:max_items]
            body = "; ".join(
                f"{t.get('ticket_id')}:{t.get('status')}/{t.get('priority')}/{t.get('category')}"
                for t in items
            )
            return f"count={payload.get('count')} [{body}]"
        if "articles" in payload and isinstance(payload["articles"], list):
            items = payload["articles"][:max_items]
            body = "; ".join(
                f"{a.get('category')}->{a.get('remedy_code')}" for a in items
            )
            return f"count={payload.get('count')} [{body}]"
        keep = {}
        for key in (
            "ticket_id",
            "status",
            "priority",
            "category",
            "remedy_code",
            "sla_hours_remaining",
            "count",
            "replies",
            "reason",
            "text",
        ):
            if key in payload:
                keep[key] = payload[key]
        if keep:
            return " ".join(f"{k}={keep[k]}" for k in keep)
    return _clip(str(payload), 220)
