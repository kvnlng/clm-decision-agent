"""Tests for the service desk example's tools, against the mock tracker."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument
import shutil
from pathlib import Path

import pytest
import service_desk_tools as desk

from clm_decision.scenario import load_scenario
from clm_decision.tools import check_tools, load_tools

EXAMPLE = Path(__file__).parent.parent / "examples" / "service_desk"


@pytest.fixture
def configured(tmp_path, tracker_url, monkeypatch):
    """The tools pointed at a copy of the example's data and at the mock tracker on a free port."""
    shutil.copy(EXAMPLE / "queues.json", tmp_path / "queues.json")
    monkeypatch.delenv("DESK_TOKEN", raising=False)
    desk.configure({"queues_file": "queues.json", "tracker_url": tracker_url, "repository": "desk-admin/requests",
                    "token_env": "DESK_TOKEN"}, tmp_path)


def request(route="Data_Access", urgency=0.4, identifiable=0.2, facts=None):
    return {"text": "Can I have the oncology registry for an audit?", "route": route,
            "answers": {"urgency": {"type": "score", "score": urgency},
                        "identifiable": {"type": "noul", "noul": identifiable}},
            "rules": [], "facts": facts or []}


def test_lookup_queue_facts(configured):
    assert desk.lookup_queue(request()) == {"facts": [
        "Data Governance handles this kind of request.",
        "It needs approval from: line manager, data steward of the dataset.",
        "The usual turnaround is 5 working days.",
        "Requests for identifiable data also need a privacy review.",
    ]}


def test_lookup_queue_without_approvals_says_none(configured):
    assert "It needs no approvals." in desk.lookup_queue(request("Information_Request"))["facts"]


def test_lookup_queue_unknown_route_has_no_facts(configured):
    assert desk.lookup_queue(request("Out_Of_Scope")) == {"facts": []}


def test_create_ticket_priority_and_label(configured, tracker_server):
    made = desk.create_ticket(request(urgency=1.7, identifiable=0.9, facts=["Data Governance handles it."]))
    issue = tracker_server.tracker.issues[made["number"]]
    assert issue["title"] == "Data_Access: Can I have the oncology registry for an audit?"
    assert "Can I have the oncology registry for an audit?" in issue["body"]
    assert "- Data Governance handles it." in issue["body"]
    assert sorted(label["name"] for label in issue["labels"]) == ["governance-review", "priority-high"]
    assert made == {"number": made["number"], "url": issue["html_url"]}


def test_create_ticket_ordinary_request(configured, tracker_server):
    made = desk.create_ticket(request())
    assert [label["name"] for label in tracker_server.tracker.issues[made["number"]]["labels"]] == \
        ["priority-medium"]


def test_create_ticket_title_is_cut_to_sixty_characters(configured, tracker_server):
    made = desk.create_ticket({**request(), "text": "word " * 30})
    assert len(tracker_server.tracker.issues[made["number"]]["title"]) == len("Data_Access: ") + 60


def test_create_ticket_tracker_down_raises(configured):
    desk.configure({**desk.SETTINGS, "tracker_url": "http://127.0.0.1:9"}, EXAMPLE)
    with pytest.raises(RuntimeError, match="could not be reached"):
        desk.create_ticket(request())


def test_example_scenario_loads(tmp_path):
    for name in ("scenario.toml", "queues.json"):
        shutil.copy(EXAMPLE / name, tmp_path / name)
    scenario = load_scenario(tmp_path / "scenario.toml")
    check_tools(scenario, load_tools(scenario))
    assert list(scenario.routes) == ["Data_Access", "Research_Support", "Information_Request", "Out_Of_Scope"]
    assert [action.tool for action in scenario.actions] == ["create_ticket"]


def test_tracker_calls_reuse_one_client(configured, tracker_server, monkeypatch):
    # A client per call costs about 17 ms on a cloud VM, more than CLM's share of a batched decision
    def no_new_client(*args, **kwargs):
        raise AssertionError("a new client was made for one call")
    monkeypatch.setattr(desk.httpx, "request", no_new_client)
    first, second = desk.create_ticket(request()), desk.create_ticket(request())
    assert second["number"] == first["number"] + 1


def test_configuring_again_closes_the_old_client(configured):
    old = desk.CLIENT
    desk.configure(dict(desk.SETTINGS), EXAMPLE)
    assert old.is_closed and not desk.CLIENT.is_closed
