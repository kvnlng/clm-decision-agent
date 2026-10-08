"""Tests for the service desk demo's tools."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,unused-argument
import json
import shutil
from pathlib import Path

import httpx
import pytest
import service_desk_tools as desk
from langchain_core.tools import ToolException

EXAMPLE = Path(__file__).parent.parent / "examples" / "service_desk"


@pytest.fixture
def configured(tmp_path, tracker_url, monkeypatch):
    """The tools pointed at copies of the demo data and at the mock tracker on a free port."""
    for name in ("queues.json", "catalogue.sql"):
        shutil.copy(EXAMPLE / name, tmp_path / name)
    monkeypatch.delenv("DESK_TOKEN", raising=False)

    def configure(**overrides):
        settings = {"queues_file": "queues.json", "catalogue_db": "catalogue.db", "catalogue_sql": "catalogue.sql",
                    "tracker_url": tracker_url, "repository": "desk-admin/requests",
                    "token_env": "DESK_TOKEN", **overrides}
        desk.configure(settings, tmp_path)
    configure()
    return configure


def test_lookup_queue_reads_the_json_file(configured):
    queue = desk.lookup_queue.invoke({"request_type": "Data_Access"})
    assert queue["team"] == "Data Governance"
    assert "data steward of the dataset" in queue["approvals"]


def test_lookup_queue_names_the_known_types_when_asked_for_another(configured):
    with pytest.raises(ToolException, match="Data_Access, Research_Support, Information_Request"):
        desk.lookup_queue.invoke({"request_type": "Billing"})


def test_configure_builds_the_catalogue_database_from_the_sql_file(configured, tmp_path):
    assert (tmp_path / "catalogue.db").exists()


def test_find_dataset_matches_name_or_description(configured):
    by_name = desk.find_dataset.invoke({"search_term": "oncology"})
    assert [row["name"] for row in by_name] == ["Oncology Registry"]
    assert by_name[0] == {"name": "Oncology Registry", "description": "Cancer diagnoses, staging and treatment dates",
                          "steward": "Cancer Services", "sensitivity": "identifiable"}
    by_description = desk.find_dataset.invoke({"search_term": "cancellations"})
    assert [row["name"] for row in by_description] == ["Theatre Utilisation"]


def test_find_dataset_uses_each_word_and_returns_at_most_five(configured):
    rows = desk.find_dataset.invoke({"search_term": "the readmissions data please"})
    assert "Readmissions Extract" in [row["name"] for row in rows]
    assert len(desk.find_dataset.invoke({"search_term": "ward"})) <= 5


@pytest.mark.parametrize("term", ["'; DROP TABLE datasets; --", 'x" OR "1"="1', "100% of %"])
def test_find_dataset_treats_odd_characters_as_text(configured, term):
    assert desk.find_dataset.invoke({"search_term": term}) == []
    assert desk.find_dataset.invoke({"search_term": "oncology"})      # the table is still there


def test_search_tickets_returns_number_summary_and_status(configured):
    found = desk.search_tickets.invoke({"text": 'access to the "readmissions" extract'})
    assert found[0]["number"] >= 1
    assert "readmissions" in found[0]["summary"].lower()
    assert found[0]["status"] == "open"


def test_create_then_get_ticket(configured, tracker_server):
    created = desk.create_ticket.invoke({"summary": "Access to the oncology registry",
                                         "description": "Six months, de-identified.",
                                         "priority": "High", "labels": ["governance-review"]})
    assert created["status"] == "created" and created["url"].endswith(f"/issues/{created['number']}")
    stored = tracker_server.tracker.issues[created["number"]]
    assert (stored["title"], stored["body"]) == ("Access to the oncology registry", "Six months, de-identified.")
    assert [label["name"] for label in stored["labels"]] == ["priority-high", "governance-review"]

    fetched = desk.get_ticket.invoke({"number": f"#{created['number']}"})
    assert fetched == {"number": created["number"], "summary": "Access to the oncology registry", "status": "open",
                       "labels": ["priority-high", "governance-review"], "url": created["url"]}


def test_a_label_the_tracker_does_not_have_is_created_once(configured, tracker_server):
    for _ in range(2):
        desk.create_ticket.invoke({"summary": "x", "description": "y", "labels": ["Needs Ethics"]})
    names = [label["name"] for label in tracker_server.tracker.labels.values()]
    assert names.count("needs-ethics") == 1


def test_get_ticket_for_an_unknown_number(configured):
    with pytest.raises(ToolException, match="No ticket with the number 9999"):
        desk.get_ticket.invoke({"number": "9999"})


@pytest.mark.parametrize("arguments, message", [
    ({"summary": " ", "description": "x"}, "A ticket needs a summary"),
    ({"summary": "x", "description": "y", "priority": "Urgent"}, "priority must be Low, Medium or High"),
])
def test_a_ticket_the_tracker_would_accept_blank_is_refused_by_the_tool(configured, tracker_server, arguments, message):
    before = len(tracker_server.tracker.issues)
    with pytest.raises(ToolException, match=message):
        desk.create_ticket.invoke(arguments)
    assert len(tracker_server.tracker.issues) == before


def test_no_credentials_are_sent_unless_the_variable_is_set(configured, tracker_server):
    desk.get_ticket.invoke({"number": "1"})
    assert tracker_server.last_authorization is None


def test_the_token_is_sent_in_the_trackers_header_form(configured, tracker_server, monkeypatch):
    monkeypatch.setenv("DESK_TOKEN", "secret")
    desk.search_tickets.invoke({"text": "readmissions"})
    assert tracker_server.last_authorization == "token secret"


def test_a_rejected_token_is_reported_plainly(configured, monkeypatch):
    monkeypatch.setattr(desk, "_request", lambda *args, **kwargs: httpx.Response(401, json={"message": "bad token"}))
    with pytest.raises(ToolException, match="rejected the credentials"):
        desk.get_ticket.invoke({"number": "1"})


def test_unreachable_tracker_says_where_and_how_to_start_it(configured):
    configured(tracker_url="http://127.0.0.1:9")
    with pytest.raises(ToolException) as failure:
        desk.create_ticket.invoke({"summary": "x", "description": "y"})
    assert "http://127.0.0.1:9" in str(failure.value)
    assert "mock_tracker.py" in str(failure.value)


def test_tools_list_names(configured):
    assert [tool.name for tool in desk.TOOLS] == [
        "lookup_queue", "find_dataset", "search_tickets", "get_ticket", "create_ticket"]
    assert json.dumps(desk.lookup_queue.invoke({"request_type": "Information_Request"}))


def test_a_failed_catalogue_build_leaves_nothing_behind_and_the_next_start_rebuilds(configured, tmp_path):
    (tmp_path / "catalogue.db").unlink()
    script = (tmp_path / "catalogue.sql").read_text()
    (tmp_path / "catalogue.sql").write_text(script + "\nINSERT INTO no_such_table VALUES (1);\n")
    with pytest.raises(Exception):
        configured()
    assert not (tmp_path / "catalogue.db").exists()
    (tmp_path / "catalogue.sql").write_text(script)
    configured()
    assert desk.find_dataset.invoke({"search_term": "oncology"})


@pytest.mark.parametrize("number", ["../labels", "1/../../admin", "1?state=all", "DSD-1", ""])
def test_get_ticket_refuses_anything_that_is_not_a_ticket_number(configured, tracker_server, number):
    tracker_server.last_authorization = "untouched"
    with pytest.raises(ToolException, match="is not a ticket number"):
        desk.get_ticket.invoke({"number": number})
    assert tracker_server.last_authorization == "untouched"
