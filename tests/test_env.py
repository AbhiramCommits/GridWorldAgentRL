from __future__ import annotations

from gwa.env import (
    CATEGORY_REMEDY,
    CRMEnv,
    ToolCall,
    execute_tool,
    generate_split,
    make_state,
    make_task,
)
from gwa.env.crm_env import _compact_payload
from gwa.env.state import make_state as make_state_direct


def make_ticket_env(family: str = "triage_and_route", seed: int = 12345, **kwargs) -> CRMEnv:
    task = make_task(family, seed)
    env = CRMEnv(task, **kwargs)
    env.reset()
    return env


# ---------------------------------------------------------------------------
# State determinism
# ---------------------------------------------------------------------------
def test_state_is_deterministic():
    a = make_state(7, 8, 20, 10)
    b = make_state(7, 8, 20, 10)
    c = make_state(8, 8, 20, 10)
    assert a.state_hash() == b.state_hash()
    assert a.state_hash() != c.state_hash()
    assert [t.ticket_id for t in a.sorted_tickets()] == [t.ticket_id for t in b.sorted_tickets()]
    assert a.tickets["T-0005"].duplicate_of == "T-0004"
    assert a.tickets["T-0007"].status == "resolved"


def test_env_reset_deterministic():
    env = make_ticket_env()
    obs1 = env.render_text()
    h1 = env.state.state_hash()
    obs2 = env.reset()
    assert obs1 == obs2
    assert env.state.state_hash() == h1


# ---------------------------------------------------------------------------
# Tool happy paths
# ---------------------------------------------------------------------------
def test_search_tickets_happy_path():
    state = make_state(1)
    res = execute_tool(state, ToolCall("search_tickets", {"status": "open"}))
    assert res.ok and res.payload["count"] >= 1
    assert all(t["status"] == "open" for t in res.payload["tickets"])

    res2 = execute_tool(state, ToolCall("search_tickets", {"query": "duplicate charge"}))
    assert res2.ok and res2.payload["count"] >= 1


def test_get_ticket_happy_path():
    state = make_state(1)
    res = execute_tool(state, ToolCall("get_ticket", {"ticket_id": "T-0001"}))
    assert res.ok
    assert res.payload["ticket_id"] == "T-0001"
    assert res.payload["customer"]["customer_id"] == "C-001"


def test_search_kb_happy_path():
    state = make_state(1)
    res = execute_tool(state, ToolCall("search_kb", {"query": "billing"}))
    assert res.ok
    assert any(a["remedy_code"] == CATEGORY_REMEDY["billing"] for a in res.payload["articles"])


def test_update_ticket_happy_path():
    state = make_state(1)
    res = execute_tool(
        state, ToolCall("update_ticket", {"ticket_id": "T-0001", "priority": "high"})
    )
    assert res.ok
    assert state.tickets["T-0001"].priority == "high"


def test_escalate_ticket_happy_path():
    state = make_state(1)
    res = execute_tool(
        state, ToolCall("escalate_ticket", {"ticket_id": "T-0002", "reason": "SLA breach"})
    )
    assert res.ok
    assert state.tickets["T-0002"].status == "escalated"
    assert state.tickets["T-0002"].escalation_reason == "SLA breach"


def test_reply_customer_happy_path():
    state = make_state(1)
    res = execute_tool(
        state, ToolCall("reply_customer", {"ticket_id": "T-0001", "message": "We are on it."})
    )
    assert res.ok
    assert state.tickets["T-0001"].replies == ["We are on it."]


# ---------------------------------------------------------------------------
# Tool failure paths (so replanning is exercised)
# ---------------------------------------------------------------------------
def test_unknown_tool_fails():
    state = make_state(1)
    res = execute_tool(state, ToolCall("delete_everything", {}))
    assert not res.ok and "unknown tool" in res.error
    assert state.invalid_actions == 1


def test_missing_required_argument_fails():
    state = make_state(1)
    res = execute_tool(state, ToolCall("escalate_ticket", {"ticket_id": "T-0002"}))
    assert not res.ok and "missing required argument" in res.error


def test_unknown_ticket_fails():
    state = make_state(1)
    res = execute_tool(state, ToolCall("get_ticket", {"ticket_id": "T-9999"}))
    assert not res.ok and "unknown ticket_id" in res.error


def test_escalate_resolved_ticket_fails():
    state = make_state(1)
    res = execute_tool(
        state, ToolCall("escalate_ticket", {"ticket_id": "T-0007", "reason": "SLA breach"})
    )
    assert not res.ok and "already resolved" in res.error


def test_wrong_remedy_code_fails():
    state = make_state(1)
    res = execute_tool(
        state,
        ToolCall(
            "update_ticket",
            {"ticket_id": "T-0001", "status": "resolved", "remedy_code": "RESET_CACHE"},
        ),
    )
    assert not res.ok and "does not match ticket category" in res.error
    assert state.tickets["T-0001"].status == "open"


def test_noop_update_fails():
    state = make_state(1)
    res = execute_tool(
        state, ToolCall("update_ticket", {"ticket_id": "T-0001", "priority": "low"})
    )
    assert not res.ok and "no-op" in res.error


# ---------------------------------------------------------------------------
# Success checks: hand-written optimal trajectories
# ---------------------------------------------------------------------------
def _run(env: CRMEnv, calls: list[ToolCall]):
    last = None
    terminated = truncated = False
    for call in calls:
        _, reward, terminated, truncated, info = env.step(call)
        last = (reward, info)
        if terminated or truncated:
            break
    return last, terminated, truncated


def test_optimal_triage_and_route():
    env = make_ticket_env("triage_and_route")
    calls = [
        ToolCall("update_ticket", {"ticket_id": t["ticket_id"], "priority": t["expected_priority"]})
        for t in env.task.spec["targets"]
    ]
    _, terminated, _ = _run(env, calls)
    ok, flags = env.success()
    assert terminated and ok
    assert all(flags.values())


def test_optimal_resolve_with_kb():
    env = make_ticket_env("resolve_with_kb")
    calls = [
        ToolCall(
            "update_ticket",
            {
                "ticket_id": t["ticket_id"],
                "status": "resolved",
                "remedy_code": t["remedy_code"],
            },
        )
        for t in env.task.spec["targets"]
    ]
    _, terminated, _ = _run(env, calls)
    ok, flags = env.success()
    assert terminated and ok, flags
    assert all(flags.values())


def test_optimal_escalate_sla_breach():
    env = make_ticket_env("escalate_sla_breach")
    calls = [
        ToolCall("escalate_ticket", {"ticket_id": tid, "reason": "SLA breach"})
        for tid in env.task.spec["targets"]
    ]
    _, terminated, _ = _run(env, calls)
    ok, flags = env.success()
    assert terminated and ok, flags
    assert flags["no_collateral_escalation"]


def test_optimal_dedupe():
    env = make_ticket_env("dedupe_duplicate_tickets")
    spec = env.task.spec
    _, terminated, _ = _run(
        env, [ToolCall("update_ticket", {"ticket_id": spec["duplicate"], "status": "resolved"})]
    )
    ok, flags = env.success()
    assert terminated and ok, flags
    assert flags["primary_untouched"]


def test_optimal_bulk_status_sweep():
    env = make_ticket_env("bulk_status_sweep")
    calls = [
        ToolCall("update_ticket", {"ticket_id": tid, "status": "pending"})
        for tid in env.task.spec["targets"]
    ]
    _, terminated, _ = _run(env, calls)
    ok, flags = env.success()
    assert terminated and ok, flags


def test_optimal_refusal():
    env = make_ticket_env("refuse_impossible_request")
    env.finish("I cannot issue a $500 refund or delete records; this violates our policy.")
    ok, flags = env.success()
    assert ok, flags
    assert flags == {"no_mutation": True, "explained_refusal": True}


# ---------------------------------------------------------------------------
# Truncation, info, and action-space surface
# ---------------------------------------------------------------------------
def test_max_steps_truncation():
    env = make_ticket_env("triage_and_route", max_steps=1)
    _, _, terminated, truncated, info = env.step(
        ToolCall("search_tickets", {"status": "open"})
    )
    assert not terminated
    assert truncated
    assert info["tool_ok"] is True
    assert info["invalid_action"] is False
    assert 0.0 <= info["progress"] <= 1.0


def test_invalid_action_info():
    env = make_ticket_env()
    _, _, _, _, info = env.step(ToolCall("get_ticket", {"ticket_id": "nope"}))
    assert info["invalid_action"] is True
    assert info["tool_ok"] is False
    assert info["error"]


def test_action_and_observation_spaces():
    env = make_ticket_env()
    assert env.action_space is not None
    assert env.observation_space is not None
    assert env.max_steps == 8
    assert len(env.tool_names) == 6


def test_compact_payload_is_short():
    payload = {
        "tickets": [
            {"ticket_id": f"T-{i:04d}", "status": "open", "priority": "low", "category": "billing"}
            for i in range(10)
        ],
        "count": 10,
    }
    text = _compact_payload(payload)
    assert "count=10" in text
    assert len(text) < 200


# ---------------------------------------------------------------------------
# Split disjointness
# ---------------------------------------------------------------------------
def test_train_eval_disjoint():
    train, evals = generate_split(12, 12, seed=0)
    train_pairs = {(t.family, t.seed) for t in train}
    eval_pairs = {(t.family, t.seed) for t in evals}
    assert not (train_pairs & eval_pairs)
    assert len(train) == 12 and len(evals) == 12
    assert {t.family for t in train} == {
        "triage_and_route",
        "resolve_with_kb",
        "escalate_sla_breach",
        "dedupe_duplicate_tickets",
        "bulk_status_sweep",
        "refuse_impossible_request",
    }


def test_task_records_roundtrip():
    task = make_task("resolve_with_kb", 999)
    restored = type(task).from_record(task.to_record())
    state = make_state_direct(task.seed, **task.env_sizes)
    assert restored.success_check(state) == task.success_check(state)
    assert restored.spec == task.spec
