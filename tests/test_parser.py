from __future__ import annotations

from gwa.agent.parser import parse_response


def test_canonical_action():
    text = (
        "PLAN:\n1. Fetch the ticket.\n2. Raise priority.\n"
        'ACTION: {"tool": "update_ticket", "args": {"ticket_id": "T-0001", "priority": "high"}}'
    )
    p = parse_response(text)
    assert not p.malformed
    assert p.plan and "Fetch" in p.plan
    assert p.action.tool == "update_ticket"
    assert p.action.args == {"ticket_id": "T-0001", "priority": "high"}


def test_markdown_fenced_json():
    text = (
        "PLAN:\n1. Look up.\n```json\n"
        'ACTION: {"tool": "get_ticket", "args": {"ticket_id": "T-0002"}}\n```'
    )
    p = parse_response(text)
    assert not p.malformed and p.action.tool == "get_ticket"


def test_single_quotes_repaired():
    text = "PLAN:\n1. Search.\nACTION: {'tool': 'search_kb', 'args': {'query': 'billing'}}"
    p = parse_response(text)
    assert not p.malformed
    assert p.action.tool == "search_kb"
    assert p.action.args["query"] == "billing"
    assert "converted single quotes" in p.repairs


def test_trailing_comma_repaired():
    text = 'PLAN:\n1. Go.\nACTION: {"tool": "get_ticket", "args": {"ticket_id": "T-0001",},}'
    p = parse_response(text)
    assert not p.malformed
    assert p.action.args["ticket_id"] == "T-0001"


def test_leading_prose_and_trailing_prose():
    text = (
        "Sure, here is my plan.\nPLAN:\n1. Search tickets.\n"
        'ACTION: {"tool": "search_tickets", "args": {"status": "open"}}\n'
        "This call should return the open tickets."
    )
    p = parse_response(text)
    assert not p.malformed
    assert p.action.tool == "search_tickets"


def test_finish_extraction():
    p = parse_response("PLAN:\n1. Nothing to do.\nFINISH: I cannot do that under policy.")
    assert p.is_finish
    assert "cannot" in p.finish


def test_finish_with_colon_in_summary():
    p = parse_response("FINISH: Declined: the request violates the $50 refund policy.")
    assert p.is_finish
    assert p.finish.startswith("Declined")


def test_missing_marker_is_malformed():
    p = parse_response("I will now update the ticket.")
    assert p.malformed
    assert "no ACTION" in p.error


def test_invalid_json_is_malformed():
    p = parse_response('PLAN:\n1. x\nACTION: {"tool": "get_ticket", "args": ')
    assert p.malformed
    assert "could not parse" in p.error


def test_missing_tool_is_malformed():
    p = parse_response('ACTION: {"args": {"ticket_id": "T-0001"}}')
    assert p.malformed
    assert "missing string 'tool'" in p.error


def test_args_not_object_is_malformed():
    p = parse_response('ACTION: {"tool": "get_ticket", "args": "T-0001"}')
    assert p.malformed
    assert "not a JSON object" in p.error


def test_nested_braces_in_string_are_safe():
    text = (
        'ACTION: {"tool": "reply_customer", "args": {"ticket_id": "T-0001", '
        '"message": "Use {curly} braces and a } here"}}'
    )
    p = parse_response(text)
    assert not p.malformed
    assert p.action.args["message"].startswith("Use {curly}")


def test_unknown_tool_passes_through_for_env_validation():
    p = parse_response('ACTION: {"tool": "drop_database", "args": {}}')
    assert not p.malformed
    assert p.action.tool == "drop_database"
    assert any("unknown tool" in r for r in p.repairs)


def test_empty_response_is_malformed():
    p = parse_response("")
    assert p.malformed and p.error


def test_action_before_finish_wins():
    text = 'ACTION: {"tool": "get_ticket", "args": {"ticket_id": "T-0001"}}\nFINISH: done'
    p = parse_response(text)
    assert p.action is not None and p.finish is None
