"""System prompt, strict response format, and token-budgeted observation formatting."""

from __future__ import annotations

from typing import Any

from ..env.tools import TOOL_NAMES, tool_manifest_text

SYSTEM_PROMPT = f"""You are an enterprise support agent operating a mock CRM.
Each turn: write a numbered PLAN, then exactly one ACTION line containing a JSON
object with keys "tool" and "args", or a FINISH line with a summary.
Rules: one ACTION per turn; valid JSON with double quotes; no markdown; no text after
the ACTION line; replan after tool errors; never invent ids or mutate resolved tickets.
Tools:
{tool_manifest_text()}
"""


def format_observation(
    obs: str,
    history: list[dict[str, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Render the current observation with a bounded window of prior turns.

    ``history`` entries are dicts with ``assistant`` (raw response), ``tool``,
    ``ok``, and ``error``. Only the last ``agent.history_turns`` entries and at
    most ``agent.history_chars`` characters are kept, so prompt growth is bounded.
    """
    cfg = cfg or {}
    agent_cfg = cfg.get("agent", cfg)
    max_turns = int(agent_cfg.get("history_turns", 3))
    max_chars = int(agent_cfg.get("history_chars", 600))
    history = history or []
    if not history:
        return f"CURRENT OBSERVATION:\n{obs}"

    chunks: list[str] = []
    for item in history[-max_turns:]:
        response = str(item.get("assistant", "")).strip()
        response = response[-max_chars:] if len(response) > max_chars else response
        status = "ok" if item.get("ok") else f"error={item.get('error')}"
        chunks.append(
            f"[turn {item.get('turn')}] assistant: {response}\n"
            f"[turn {item.get('turn')}] tool {item.get('tool')} -> {status}"
        )
    hist_text = "\n".join(chunks)
    if len(hist_text) > max_chars * 2:
        hist_text = hist_text[-(max_chars * 2) :]
    return f"RECENT HISTORY:\n{hist_text}\n\nCURRENT OBSERVATION:\n{obs}"


def build_user_content(
    task_instruction: str,
    obs: str,
    history: list[dict[str, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> str:
    """The per-turn user message sent to a policy backend."""
    # The observation already starts with the task instruction (env.render_text),
    # so do not duplicate it here: prompt length matters on a CPU build.
    return format_observation(obs, history, cfg)


def example_response() -> str:
    """A canonical well-formed response, used in tests and documentation."""
    return (
        "PLAN:\n"
        "1. Look up the ticket.\n"
        "2. Update its priority.\n"
        'ACTION: {"tool": "update_ticket", "args": {"ticket_id": "T-0001", '
        '"priority": "high"}}\n'
    )


def available_tools() -> tuple[str, ...]:
    return TOOL_NAMES
