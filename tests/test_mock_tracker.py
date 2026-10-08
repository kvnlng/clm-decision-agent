"""Tests for the mock tracker in examples/service_desk."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
import httpx

ISSUES = "/api/v1/repos/desk-admin/requests/issues"
LABELS = "/api/v1/repos/desk-admin/requests/labels"


def test_create_then_fetch(tracker_url):
    label = httpx.post(tracker_url + LABELS, json={"name": "governance-review", "color": "#5319e7"})
    assert label.status_code == 201
    created = httpx.post(tracker_url + ISSUES, json={
        "title": "Access to the oncology registry", "body": "Six months, de-identified.",
        "labels": [label.json()["id"]]})
    assert created.status_code == 201
    body = created.json()
    assert body["number"] >= 1 and body["html_url"].endswith(f"/desk-admin/requests/issues/{body['number']}")

    fetched = httpx.get(f"{tracker_url}{ISSUES}/{body['number']}")
    assert fetched.status_code == 200
    issue = fetched.json()
    assert (issue["title"], issue["body"], issue["state"]) == \
        ("Access to the oncology registry", "Six months, de-identified.", "open")
    assert [found["name"] for found in issue["labels"]] == ["governance-review"]


def test_labels_given_as_names_are_refused_like_forgejo(tracker_url):
    response = httpx.post(tracker_url + ISSUES, json={"title": "x", "labels": ["governance-review"]})
    assert response.status_code == 422
    assert "labels" in response.json()["message"]


def test_body_that_is_not_json_is_a_422(tracker_url):
    assert httpx.post(tracker_url + ISSUES, content=b"not json").status_code == 422


def test_unknown_ticket_is_a_404(tracker_url):
    response = httpx.get(f"{tracker_url}{ISSUES}/9999")
    assert response.status_code == 404
    assert response.json()["message"] == "The target couldn't be found."


def test_search_finds_seeded_tickets_by_any_word(tracker_url):
    params = {"q": "readmissions zebra", "state": "all", "type": "issues", "limit": 3}
    found = httpx.get(tracker_url + ISSUES, params=params).json()
    assert 1 <= len(found) <= 3
    assert "readmissions" in found[0]["title"].lower()


def test_search_with_no_match_returns_no_issues(tracker_url):
    assert httpx.get(tracker_url + ISSUES, params={"q": "zebra xylophone"}).json() == []


def test_labels_are_listed_and_a_repeated_name_is_accepted_like_forgejo(tracker_url):
    names = [label["name"] for label in httpx.get(tracker_url + LABELS).json()]
    assert {"priority-high", "priority-medium", "priority-low"} <= set(names)
    assert httpx.post(tracker_url + LABELS, json={"name": "priority-high", "color": "#000000"}).status_code == 201
    assert [label["name"] for label in httpx.get(tracker_url + LABELS).json()].count("priority-high") == 2


def test_the_server_notes_the_authorization_header(tracker_server, tracker_url):
    httpx.get(f"{tracker_url}{ISSUES}/1", headers={"Authorization": "token abc"})
    assert tracker_server.last_authorization == "token abc"


def test_unknown_path_is_a_404(tracker_url):
    assert httpx.get(f"{tracker_url}/api/v1/nothing").status_code == 404
