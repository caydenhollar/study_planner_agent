from copy import deepcopy
import json
from pathlib import Path

import pytest

from planner import plan, render_report


def task(id="a", due="2026-11-24", minutes=60, priority=3):
    return {"id": id, "title": id, "due": due,
            "minutes": minutes, "priority": priority}


def test_earlier_deadline_beats_priority():
    result = plan([task("later", due="2026-11-25", priority=5),
                   task("urgent", priority=1)], {"2026-11-23": 60})
    assert {s["task_id"] for s in result["sessions"]} == {"urgent"}
    assert result["unplanned"][0]["task_id"] == "later"


def test_priority_breaks_deadline_tie():
    result = plan([task("low", priority=1), task("high", priority=5)],
                  {"2026-11-23": 30})
    assert result["sessions"][0]["task_id"] == "high"


def test_unfinished_work_carries_to_next_day():
    result = plan([task(minutes=75)], {"2026-11-23": 40, "2026-11-24": 40})
    assert [s["minutes"] for s in result["sessions"]] == [30, 10, 30, 5]
    assert result["unused_minutes"]["2026-11-24"] == 5
    assert result["unplanned"] == []


def test_expired_task_is_reported_not_scheduled():
    result = plan([task(due="2026-11-22")], {"2026-11-23": 120})
    assert result["sessions"] == []
    assert result["unplanned"][0]["minutes_left"] == 60


def test_example_respects_deadlines_and_capacity():
    data = json.loads((Path(__file__).parents[1] / "examples/week.json").read_text())
    original = deepcopy(data)
    result = plan(data["tasks"], data["availability"])
    tasks = {t["id"]: t for t in data["tasks"]}
    for day, capacity in data["availability"].items():
        assert sum(s["minutes"] for s in result["sessions"] if s["day"] == day) <= capacity
    for t in tasks.values():
        sessions = [s for s in result["sessions"] if s["task_id"] == t["id"]]
        remaining = sum(u["minutes_left"] for u in result["unplanned"]
                        if u["task_id"] == t["id"])
        assert sum(s["minutes"] for s in sessions) + remaining == t["minutes"]
        assert all(s["day"] <= t["due"] for s in sessions)
    assert data == original
    assert result["framework"] == "LangGraph"
    assert result["trace"][0]["node"] == "inspect_day"
    assert "minutes left" in render_report(result)


def test_model_cannot_book_an_expired_task():
    tasks = [task("expired", due="2026-11-22"), task("valid")]
    with pytest.raises(ValueError):
        plan(tasks, {"2026-11-23": 20},
             choose=lambda *args: {"task_id": "expired", "reason": "Try late task"})


@pytest.mark.parametrize("bad", [task(minutes=-1), task(priority=6),
                                 task(due="not-a-date"), task(minutes=True),
                                 task(due="20261124")])
def test_invalid_tasks(bad):
    with pytest.raises(ValueError):
        plan([bad], {"2026-11-23": 60})


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        plan([task(), task()], {})


def test_invalid_capacity_and_session_size():
    with pytest.raises(ValueError):
        plan([task()], {"2026-11-23": -1})
    with pytest.raises(ValueError):
        plan([task()], {"2026-11-23": 60}, chunk=0)


def test_empty_week():
    result = plan([], {"2026-11-23": 0})
    assert result["sessions"] == []
    assert "Everything fits" in render_report(result)


def test_decisions_see_updated_state_after_every_booking():
    observations = []

    def choose(day, free, candidates, context):
        observations.append((free, candidates[0]["minutes_left"]))
        return {"task_id": "a", "reason": "Continue the assignment",
                "study_goal": "Work through the assignment", "source_ids": []}

    result = plan([task()], {"2026-11-23": 75}, choose=choose)
    assert observations == [(75, 60), (45, 30)]
    assert sum(s["minutes"] for s in result["sessions"]) == 60
    assert result["unused_minutes"]["2026-11-23"] == 15


def test_model_choice_controls_the_schedule():
    result = plan([task("a"), task("b")], {"2026-11-23": 30},
                  choose=lambda *args: {"task_id": "b", "reason": "Prepare for group work",
                                        "study_goal": "Work through assignment b", "source_ids": []})
    assert result["sessions"][0]["task_id"] == "b"
    assert any(t.get("reason") == "Prepare for group work" for t in result["trace"])


@pytest.mark.parametrize("choice", [None, {}, {"task_id": "missing", "reason": "Unknown"},
                                     {"task_id": "a", "reason": ""},
                                     {"task_id": "a", "reason": 5}])
def test_invalid_model_decisions_are_rejected(choice):
    with pytest.raises(ValueError, match="eligible task"):
        plan([task()], {"2026-11-23": 30}, choose=lambda *args: choice)


def test_many_sessions_finish_beyond_default_graph_recursion_limit():
    result = plan([task(minutes=60)], {"2026-11-23": 60}, chunk=1)
    assert len(result["sessions"]) == 60
    assert result["unplanned"] == []


def test_no_availability_finishes_without_calling_model():
    def unexpected_call(*args):
        raise AssertionError("No model call should be needed")

    result = plan([task()], {}, choose=unexpected_call)
    assert result["sessions"] == []
    assert result["unplanned"][0]["minutes_left"] == 60
