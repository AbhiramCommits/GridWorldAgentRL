"""Robust parser for the PLAN / ACTION / FINISH response format.

LLM output is messy: markdown fences, trailing prose, single quotes, trailing commas,
and unescaped control characters all occur in practice. The parser repairs what it can
and marks the rest ``malformed`` so the loop can feed an error back and count a REPLAN.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..env.tools import TOOL_NAMES, ToolCall

_PLAN_RE = re.compile(r"PLAN\s*:\s*(.*?)(?=\n\s*(?:ACTION\s*:|FINISH\s*:)|$)", re.S | re.I)
_ACTION_RE = re.compile(r"ACTION\s*:", re.I)
_FINISH_RE = re.compile(r"FINISH\s*:\s*(.*)", re.S | re.I)


@dataclass
class ParsedAction:
    plan: str | None = None
    action: ToolCall | None = None
    finish: str | None = None
    malformed: bool = False
    error: str | None = None
    raw: str = ""
    repairs: list[str] = field(default_factory=list)

    @property
    def is_finish(self) -> bool:
        return self.finish is not None and self.action is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "action": self.action.to_dict() if self.action else None,
            "finish": self.finish,
            "malformed": self.malformed,
            "error": self.error,
            "repairs": self.repairs,
        }


def parse_response(text: str) -> ParsedAction:
    """Parse one model turn into a plan plus an ACTION, a FINISH, or malformed."""
    raw = text or ""
    cleaned = _strip_fences(raw)
    plan_match = _PLAN_RE.search(cleaned)
    plan = _normalize_plan(plan_match.group(1)) if plan_match else None

    action_match = _ACTION_RE.search(cleaned)
    finish_match = _FINISH_RE.search(cleaned)
    action_pos = action_match.start() if action_match else 10**9
    finish_pos = finish_match.start() if finish_match else 10**9

    if action_pos == 10**9 and finish_pos == 10**9:
        return ParsedAction(
            plan=plan,
            malformed=True,
            error="no ACTION: or FINISH: found",
            raw=raw,
        )

    # Whichever marker appears first wins; the strict format allows exactly one.
    if finish_pos < action_pos:
        summary = _clean_finish(finish_match.group(1)) if finish_match else ""
        return ParsedAction(plan=plan, finish=summary, raw=raw)

    if action_match is None:
        return ParsedAction(plan=plan, malformed=True, error="missing ACTION payload", raw=raw)

    start = cleaned.find("{", action_match.end())
    if start == -1:
        return ParsedAction(
            plan=plan, malformed=True, error="ACTION has no JSON object", raw=raw
        )
    json_text, repairs = _extract_balanced(cleaned, start)
    obj, err = _loads_tolerant(json_text, repairs)
    if obj is None:
        return ParsedAction(
            plan=plan,
            malformed=True,
            error=f"could not parse ACTION JSON: {err}",
            raw=raw,
            repairs=repairs,
        )
    if not isinstance(obj, dict):
        return ParsedAction(
            plan=plan, malformed=True, error="ACTION JSON is not an object", raw=raw, repairs=repairs
        )
    tool = obj.get("tool") or obj.get("name") or obj.get("tool_name")
    args = obj.get("args") or obj.get("arguments") or {}
    if not isinstance(tool, str) or not tool.strip():
        return ParsedAction(
            plan=plan, malformed=True, error="ACTION missing string 'tool'", raw=raw, repairs=repairs
        )
    if not isinstance(args, dict):
        return ParsedAction(
            plan=plan,
            malformed=True,
            error="ACTION 'args' is not a JSON object",
            raw=raw,
            repairs=repairs,
        )
    tool = tool.strip()
    if tool not in TOOL_NAMES:
        # Not malformed at the parser level: the environment returns a realistic
        # "unknown tool" error and the agent gets a chance to replan.
        repairs.append(f"unknown tool {tool!r} passed through for env validation")
    return ParsedAction(
        plan=plan,
        action=ToolCall(tool=tool, args=args),
        raw=raw,
        repairs=repairs,
    )


# ---------------------------------------------------------------------------
# Repair helpers
# ---------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    text = re.sub(r"```(?:json|text)?\s*", "", text)
    return text.replace("```", "")


def _normalize_plan(plan: str) -> str:
    lines = [ln.strip() for ln in plan.strip().splitlines() if ln.strip()]
    return "\n".join(lines)


def _clean_finish(summary: str) -> str:
    summary = summary.strip()
    summary = re.sub(r"^[\s\-*]+", "", summary)
    return summary[:800]


def _extract_balanced(text: str, start: int) -> tuple[str, list[str]]:
    """Return the balanced JSON object starting at ``start`` and any repairs."""
    depth = 0
    in_string = False
    escape = False
    end = len(text)
    for i in range(start, len(text)):
        ch = text[i]
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
                end = i + 1
                break
    return text[start:end], []


def _loads_tolerant(text: str, repairs: list[str]) -> tuple[Any, str | None]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as first_err:
        candidates: list[tuple[str, str]] = []
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        candidates.append((fixed, "removed trailing commas"))
        if "'" in fixed and '"' not in fixed:
            candidates.append((fixed.replace("'", '"'), "converted single quotes"))
        candidates.append(
            (re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", fixed)), "python bools")
        )
        for cand, note in candidates:
            try:
                obj = json.loads(cand)
                if note not in repairs:
                    repairs.append(note)
                return obj, None
            except json.JSONDecodeError:
                continue
        try:
            obj = ast.literal_eval(text)
            repairs.append("parsed with ast.literal_eval")
            return obj, None
        except (ValueError, SyntaxError):
            pass
        return None, str(first_err)
