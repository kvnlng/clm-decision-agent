"""Tools for the data service desk example. The decision server calls each one with the request:
{"text", "route", "answers", "rules", "facts"}, filled in as far as the decision has got.

    lookup_queue    an informing tool: who handles the route, from a local JSON file
    create_ticket   an acting tool: opens a ticket in an issue tracker, through its REST API

The tracker calls follow the REST API of Forgejo and Gitea, which the mock tracker imitates.
The scenario's [tools.settings] table is passed to configure(): queues_file, tracker_url, repository
("owner/name") and token_env, the name of the environment variable that holds an access token.
For another tracker, rewrite create_ticket and its helpers; nothing else depends on them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

SETTINGS: dict = {}
PATH_SETTINGS = ("queues_file",)
TITLE_LENGTH = 60


class TrackerError(RuntimeError):
    """The tracker could not be reached or refused a request; the message says which."""


def configure(settings: dict, scenario_dir: Path) -> None:
    """Take the scenario's settings, with file names taken as relative to the scenario's folder."""
    SETTINGS.clear()
    SETTINGS.update(settings)
    for key in PATH_SETTINGS:
        if key in SETTINGS:
            SETTINGS[key] = str(Path(scenario_dir) / SETTINGS[key])


def lookup_queue(request: dict) -> dict:
    """Facts about the team that handles the request's route: who, which approvals, how long."""
    queues = json.loads(Path(SETTINGS["queues_file"]).read_text(encoding="utf-8"))
    queue = queues.get(request["route"])
    if queue is None:
        return {"facts": []}
    approvals = ", ".join(queue["approvals"])
    facts = [f"{queue['team']} handles this kind of request.",
             f"It needs approval from: {approvals}." if approvals else "It needs no approvals.",
             f"The usual turnaround is {queue['turnaround']}."]
    return {"facts": facts + ([queue["note"]] if queue.get("note") else [])}


def _request(method: str, path: str, **kwargs) -> httpx.Response:
    """One call to the tracker's REST API for the configured repository."""
    headers = {"Accept": "application/json"}
    token = os.environ.get(SETTINGS.get("token_env", ""), "")
    if token:
        headers["Authorization"] = f"token {token}"
    base = SETTINGS["tracker_url"].rstrip("/")
    try:
        return httpx.request(method, f"{base}/api/v1/repos/{SETTINGS['repository']}{path}",
                             headers=headers, timeout=10, **kwargs)
    except httpx.HTTPError as error:
        raise TrackerError(
            f"The tracker at {base} could not be reached ({type(error).__name__}). "
            f"For the example, start it with: uv run python examples/service_desk/mock_tracker.py") from error


def _checked(response: httpx.Response, action: str, expected: int = 200):
    """The response's JSON, or a TrackerError that says what the tracker answered."""
    if response.status_code in (401, 403):
        raise TrackerError(f"The tracker rejected the credentials (HTTP {response.status_code}). "
                           f"Check the token in the variable named by token_env.")
    if response.status_code != expected:
        try:
            reason = response.json().get("message") or f"HTTP {response.status_code}"
        except (ValueError, AttributeError):
            reason = f"HTTP {response.status_code}"
        raise TrackerError(f"The tracker could not {action}: {reason}")
    return response.json()


def _label_ids(names: list[str]) -> list[int]:
    """The tracker's ids for these label names; a label it does not have yet is created."""
    existing = {label["name"]: label["id"] for label in _checked(_request("GET", "/labels"), "list its labels")}
    ids = []
    for name in names:
        if name not in existing:
            # The tracker would accept a second label of the same name, so only create what is missing
            created = _checked(_request("POST", "/labels", json={"name": name, "color": "#888888"}),
                               "create a label", 201)
            existing[name] = created["id"]
        ids.append(existing[name])
    return ids


def _value(answers: dict, name: str) -> float:
    answer = answers.get(name) or {}
    return answer.get(answer.get("type"), 0.0)


def create_ticket(request: dict) -> dict:
    """Open a ticket for the request. Priority and the governance label come from CLM's answers.

    There is no language model to reword the request, so the ticket holds the requester's own words."""
    text = " ".join(request["text"].split())
    labels = ["priority-high" if _value(request["answers"], "urgency") >= 1.5 else "priority-medium"]
    if _value(request["answers"], "identifiable") >= 0.5:
        labels.append("governance-review")
    body = f"{text}\n\nKnown:\n" + "\n".join(f"- {fact}" for fact in request["facts"]) if request["facts"] else text
    created = _checked(_request("POST", "/issues", json={"title": f"{request['route']}: {text[:TITLE_LENGTH]}",
                                                         "body": body, "labels": _label_ids(labels)}),
                       "create the ticket", 201)
    return {"number": created["number"], "url": created.get("html_url")}


TOOLS = [lookup_queue, create_ticket]
