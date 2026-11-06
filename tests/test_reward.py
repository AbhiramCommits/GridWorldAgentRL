from __future__ import annotations

import pytest

from gwa.env import CRMEnv, ToolCall, dense_reward, episode_return, make_state, make_task
from gwa.env.reward import RewardConfig, terminal_reward


def _triage_setup():
    task = make_task("triage_and_route", 12345)
    target = task.spec["targets"][0]
    state = make_state(task.seed, **task.env_sizes)
    state.tickets[target["ticket_id"]].priority = target["expected_priority"]
    return task, state, target


def test_terminal_reward_success_and_invalid_penalty():
    task, state, _ = _triage_setup()
    cfg = RewardConfig(invalid_action_penalty=0.1)
    assert terminal_reward(task, state, cfg) == pytest.approx(1.0)
    state.invalid_actions = 2
    assert terminal_reward(task, state, cfg) == pytest.approx(0.8)


def test_terminal_reward_failure_is_zero():
    task = make_task("triage_and_route", 12345)
    state = make_state(task.seed, **task.env_sizes)
    assert terminal_reward(task, state) == 0.0


def test_dense_valid_action_beats_invalid_action():
    task, _, target = _triage_setup()
    env = CRMEnv(task)
    env.reset()
    _, _, _, _, good_info = env.step(
        ToolCall(
            "update_ticket",
            {"ticket_id": target["ticket_id"], "priority": target["expected_priority"]},
        )
    )
    good = dense_reward(task, env.state, good_info, 1)

    bad_env = CRMEnv(task)
    bad_env.reset()
    _, _, _, _, bad_info = bad_env.step(ToolCall("get_ticket", {"ticket_id": "T-9999"}))
    bad = dense_reward(task, bad_env.state, bad_info, 1)
    assert good > bad
    assert bad < 0


def test_dense_credits_new_subgoal_only_once():
    task, _, target = _triage_setup()
    env = CRMEnv(task)
    env.reset()
    _, _, _, _, info = env.step(
        ToolCall(
            "update_ticket",
            {"ticket_id": target["ticket_id"], "priority": target["expected_priority"]},
        )
    )
    first = dense_reward(task, env.state, info, 1)
    # A second valid but already-satisfied step has no new subgoal credit.
    _, _, _, _, info2 = env.step(ToolCall("search_tickets", {"status": "open"}))
    second = dense_reward(task, env.state, info2, 2)
    assert first > second


def test_sparse_vs_dense_return_shape():
    assert episode_return("sparse", [0.2, 0.3], 0.0) == 0.0
    assert episode_return("dense", [0.2, 0.3], 0.0) == pytest.approx(0.5)
    assert episode_return("dense", [0.2], 1.0) == pytest.approx(1.2)


def test_unknown_reward_mode_raises():
    with pytest.raises(ValueError):
        episode_return("banana", [], 0.0)
