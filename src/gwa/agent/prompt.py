"""System prompt, strict response format, and token-budgeted observation formatting."""

from __future__ import annotations

from typing import Any

from ..env.tools import TOOL_NAMES, tool_manifest_text

SYSTEM_PROMPT = f"""You are an enterprise support agent operating a mock CRM.
You plan, call exactly one tool per turn, observe the result, and replan when a tool fails.

STRICT RESPONSE FORMAT. Every turn must be either:

PLAN:
1. <step>
2. <step>
ACTION: {{"tool": "<tool_name>", "args": {{...}}}}

or, when the task is complete or must be refused:

FINISH: <one-sentence summary>

Rules:
- Emit at most one ACTION per turn, on one line, as valid JSON with double quotes.
- The plan is numbered and comes before ACTION.
- Use only these tools:
{tool_manifest_text()}
- Never invent ticket ids. Never mutate resolved tickets.
- If a tool returns an error, change your plan and try a corrected action.
- Do not wrap JSON in markdown fences and do not add text after the ACTION line.
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
    body = format_observation(obs, history, cfg)
    return f"TASK INSTRUCTION:\n{task_instruction}\n\n{body}"


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
