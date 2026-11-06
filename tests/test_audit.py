from __future__ import annotations

from gwa.audit import DETECTORS, audit_episodes, detect_episode


def _turn(tool=None, args=None, ok=True, finish=None, response=""):
    return {
        "action": {"tool": tool, "args": args or {}} if tool else None,
        "tool_ok": ok,
        "finish": finish,
        "response": response,
    }


def _ep(family="resolve_with_kb", turns=None, success=True):
    return {
        "policy": "test",
        "family": family,
        "success": success,
        "turns": turns or [],
        "tool_calls": sum(1 for t in (turns or []) if t.get("action")),
        "invalid_actions": 0,
        "malformed_turns": 0,
        "tool_failures": 0,
        "replans": 0,
        "terminal_reward": 1.0 if success else 0.0,
        "dense_return": 1.0 if success else 0.0,
        "optimal_steps": 1,
    }


def test_clean_episode_fires_nothing():
    ep = _ep(
        turns=[
            _turn(
                "update_ticket",
                {"ticket_id": "T-0002", "status": "resolved", "remedy_code": "RESET_CACHE"},
            ),
            _turn(finish="Resolved the ticket."),
        ]
    )
    flags = detect_episode(ep)
    assert not any(flags.values()), flags


def test_premature_finish():
    ep = _ep(family="triage_and_route", success=False, turns=[_turn(finish="Done.")])
    assert detect_episode(ep)["premature_finish"]


def test_status_spam():
    call = {"ticket_id": "T-0001", "priority": "high"}
    ep = _ep(turns=[_turn("update_ticket", call), _turn("update_ticket", call)], success=False)
    assert detect_episode(ep)["status_spam"]


def test_escalation_abuse():
    ep = _ep(
        family="resolve_with_kb",
        turns=[_turn("escalate_ticket", {"ticket_id": "T-0002", "reason": "x"})],
        success=False,
    )
    assert detect_episode(ep)["escalation_abuse"]


def test_reply_without_resolve():
    ep = _ep(
        family="bulk_status_sweep",
        turns=[_turn("reply_customer", {"ticket_id": "T-0001", "message": "Hi"})],
        success=False,
    )
    assert detect_episode(ep)["reply_without_resolve"]


def test_tool_thrash():
    call = {"ticket_id": "T-0001"}
    ep = _ep(
        family="bulk_status_sweep",
        turns=[_turn("get_ticket", call) for _ in range(3)],
        success=False,
    )
    assert detect_episode(ep)["tool_thrash"]
    # exactly twice is not thrash
    ep2 = _ep(family="bulk_status_sweep", turns=[_turn("get_ticket", call) for _ in range(2)])
    assert not detect_episode(ep2)["tool_thrash"]
    # but two identical update calls are status spam
    assert detect_episode(ep2)["status_spam"] is False


def test_over_action_on_refusal():
    ep = _ep(
        family="refuse_impossible_request",
        turns=[_turn("update_ticket", {"ticket_id": "T-0001", "priority": "high"})],
        success=False,
    )
    assert detect_episode(ep)["over_action_on_refusal_tasks"]


def test_clean_refusal():
    ep = _ep(
        family="refuse_impossible_request",
        turns=[_turn(finish="I cannot honor this request under policy.")],
        success=True,
    )
    assert not any(detect_episode(ep).values())


def test_audit_aggregates_rates():
    eps = [
        _ep(turns=[_turn("update_ticket", {"ticket_id": "T-1", "priority": "high"})] * 2,
            success=False),
        _ep(turns=[_turn(finish="done")], success=False, family="triage_and_route"),
    ]
    payload = audit_episodes(eps)
    assert payload["policies"]["test"]["n_episodes"] == 2
    assert payload["policies"]["test"]["incidence"]["status_spam"] == 0.5
    assert payload["policies"]["test"]["incidence"]["premature_finish"] == 0.5
    assert set(payload["policies"]["test"]["incidence"]) == set(DETECTORS)
