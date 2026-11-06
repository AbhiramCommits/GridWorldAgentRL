"""The six validated CRM tools exposed to the agent.

Every tool returns a :class:`ToolResult`. Validation is intentionally strict and
realistic (unknown ids, missing arguments, illegal transitions, wrong remedy codes)
so that the agent must actually replan after failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .state import CATEGORIES, CATEGORY_REMEDY, PRIORITIES, STATUSES, CRMState

TOOL_NAMES: tuple[str, ...] = (
    "search_tickets",
    "get_ticket",
    "search_kb",
    "update_ticket",
    "escalate_ticket",
    "reply_customer",
)

MUTATING_TOOLS: frozenset[str] = frozenset({"update_ticket", "escalate_ticket", "reply_customer"})

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_tickets": {
        "description": "Search tickets by free-text query, customer id, and/or status.",
        "required": [],
        "optional": ["query", "customer_id", "status"],
    },
    "get_ticket": {
        "description": "Fetch one ticket by id.",
        "required": ["ticket_id"],
        "optional": [],
    },
    "search_kb": {
        "description": "Search the knowledge base by category, remedy code, or text.",
        "required": ["query"],
        "optional": [],
    },
    "update_ticket": {
        "description": "Update status, priority, and/or remedy code on a ticket.",
        "required": ["ticket_id"],
        "optional": ["status", "priority", "remedy_code"],
    },
    "escalate_ticket": {
        "description": "Escalate an unresolved ticket with a reason.",
        "required": ["ticket_id", "reason"],
        "optional": [],
    },
    "reply_customer": {
        "description": "Append a reply message to a ticket.",
        "required": ["ticket_id", "message"],
        "optional": [],
    },
}


@dataclass
class ToolResult:
    ok: bool
    payload: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "payload": self.payload, "error": self.error}


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(tool=str(data.get("tool", "")), args=dict(data.get("args") or {}))


def _ticket_summary(state: CRMState, ticket_id: str) -> dict[str, Any]:
    t = state.tickets[ticket_id]
    return {
        "ticket_id": t.ticket_id,
        "customer_id": t.customer_id,
        "status": t.status,
        "priority": t.priority,
        "category": t.category,
        "sla_hours_remaining": t.sla_hours_remaining,
        "text": t.text,
        "remedy_code": t.remedy_code,
        "duplicate_of": t.duplicate_of,
        "replies": list(t.replies),
    }


def _err(msg: str) -> ToolResult:
    return ToolResult(ok=False, payload=None, error=msg)


def execute_tool(state: CRMState, call: ToolCall) -> ToolResult:
    """Validate and execute one tool call, mutating ``state`` on success.

    The action log, tool-call counter, and invalid-action counter are updated here so
    both the environment and the reward-hacking audit see every attempt.
    """
    state.tool_calls += 1
    name = call.tool
    args = call.args

    if name not in TOOL_SCHEMAS:
        result = _err(f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}")
        _log(state, call, result)
        return result
    if not isinstance(args, dict):
        result = _err("args must be a JSON object")
        _log(state, call, result)
        return result

    missing = [k for k in TOOL_SCHEMAS[name]["required"] if k not in args or args[k] in (None, "")]
    if missing:
        result = _err(f"missing required argument(s): {', '.join(missing)}")
        _log(state, call, result)
        return result

    handler = {
        "search_tickets": _search_tickets,
        "get_ticket": _get_ticket,
        "search_kb": _search_kb,
        "update_ticket": _update_ticket,
        "escalate_ticket": _escalate_ticket,
        "reply_customer": _reply_customer,
    }[name]
    result = handler(state, args)
    _log(state, call, result)
    return result


def _log(state: CRMState, call: ToolCall, result: ToolResult) -> None:
    if not result.ok:
        state.invalid_actions += 1
    state.action_log.append(
        {
            "turn": state.current_turn,
            "tool": call.tool,
            "args": call.args,
            "ok": result.ok,
            "error": result.error,
        }
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _search_tickets(state: CRMState, args: dict[str, Any]) -> ToolResult:
    status = args.get("status")
    if status is not None and status not in STATUSES:
        return _err(f"invalid status {status!r}; expected one of {STATUSES}")
    customer_id = args.get("customer_id")
    if customer_id is not None and customer_id not in state.customers:
        return _err(f"unknown customer_id {customer_id!r}")
    query = args.get("query")
    q = str(query).lower() if query is not None else None

    matches = []
    for t in state.sorted_tickets():
        if status is not None and t.status != status:
            continue
        if customer_id is not None and t.customer_id != customer_id:
            continue
        if q is not None and q not in (t.text + " " + t.ticket_id + " " + t.category).lower():
            continue
        matches.append(_ticket_summary(state, t.ticket_id))
    return ToolResult(ok=True, payload={"count": len(matches), "tickets": matches[:20]})


def _get_ticket(state: CRMState, args: dict[str, Any]) -> ToolResult:
    tid = str(args["ticket_id"])
    if tid not in state.tickets:
        return _err(f"unknown ticket_id {tid!r}")
    payload = _ticket_summary(state, tid)
    payload["customer"] = state.customers[state.tickets[tid].customer_id].__dict__
    return ToolResult(ok=True, payload=payload)


def _search_kb(state: CRMState, args: dict[str, Any]) -> ToolResult:
    q = str(args["query"]).strip().lower()
    if not q:
        return _err("query must be non-empty")
    matches = []
    for a in state.articles:
        blob = f"{a.article_id} {a.category} {a.remedy_code} {a.title} {a.body}".lower()
        if q in blob or (q in CATEGORIES and a.category == q):
            matches.append(
                {
                    "article_id": a.article_id,
                    "category": a.category,
                    "remedy_code": a.remedy_code,
                    "title": a.title,
                }
            )
    if not matches:
        return _err(f"no KB article matched {q!r}")
    return ToolResult(ok=True, payload={"count": len(matches), "articles": matches})


def _update_ticket(state: CRMState, args: dict[str, Any]) -> ToolResult:
    tid = str(args["ticket_id"])
    if tid not in state.tickets:
        return _err(f"unknown ticket_id {tid!r}")
    status = args.get("status")
    priority = args.get("priority")
    remedy_code = args.get("remedy_code")
    if status is None and priority is None and remedy_code is None:
        return _err("no fields to update; provide status, priority, and/or remedy_code")
    if status is not None and status not in STATUSES:
        return _err(f"invalid status {status!r}; expected one of {STATUSES}")
    if priority is not None and priority not in PRIORITIES:
        return _err(f"invalid priority {priority!r}; expected one of {PRIORITIES}")

    ticket = state.tickets[tid]
    if ticket.status == "resolved":
        return _err(f"ticket {tid} is already resolved and cannot be updated")
    if remedy_code is not None:
        expected = CATEGORY_REMEDY[ticket.category]
        if remedy_code != expected:
            return _err(
                f"remedy_code {remedy_code!r} does not match ticket category "
                f"{ticket.category!r} (expected {expected!r})"
            )
    no_change = (
        (status is None or status == ticket.status)
        and (priority is None or priority == ticket.priority)
        and (remedy_code is None or remedy_code == ticket.remedy_code)
    )
    if no_change:
        return _err(f"no-op update on {tid}: requested values already set")

    if status is not None:
        ticket.status = status
    if priority is not None:
        ticket.priority = priority
    if remedy_code is not None:
        ticket.remedy_code = remedy_code
    ticket.update_count += 1
    return ToolResult(
        ok=True,
        payload={
            "ticket_id": tid,
            "status": ticket.status,
            "priority": ticket.priority,
            "remedy_code": ticket.remedy_code,
        },
    )


def _escalate_ticket(state: CRMState, args: dict[str, Any]) -> ToolResult:
    tid = str(args["ticket_id"])
    reason = str(args.get("reason") or "").strip()
    if not reason:
        return _err("reason must be non-empty")
    if tid not in state.tickets:
        return _err(f"unknown ticket_id {tid!r}")
    ticket = state.tickets[tid]
    if ticket.status == "resolved":
        return _err(f"cannot escalate {tid}: ticket is already resolved")
    if ticket.status == "escalated":
        return _err(f"ticket {tid} is already escalated")
    ticket.status = "escalated"
    ticket.escalation_reason = reason
    ticket.update_count += 1
    return ToolResult(ok=True, payload={"ticket_id": tid, "status": "escalated", "reason": reason})


def _reply_customer(state: CRMState, args: dict[str, Any]) -> ToolResult:
    tid = str(args["ticket_id"])
    message = str(args.get("message") or "").strip()
    if not message:
        return _err("message must be non-empty")
    if len(message) > 500:
        return _err("message too long (max 500 chars)")
    if tid not in state.tickets:
        return _err(f"unknown ticket_id {tid!r}")
    state.tickets[tid].replies.append(message)
    return ToolResult(ok=True, payload={"ticket_id": tid, "replies": len(state.tickets[tid].replies)})


def tool_manifest_text() -> str:
    """Compact tool documentation embedded in the system prompt."""
    lines = []
    for name in TOOL_NAMES:
        schema = TOOL_SCHEMAS[name]
        req = ", ".join(schema["required"]) or "none"
        opt = ", ".join(schema["optional"]) or "none"
        lines.append(f"- {name}: required=[{req}] optional=[{opt}] - {schema['description']}")
    return "\n".join(lines)
