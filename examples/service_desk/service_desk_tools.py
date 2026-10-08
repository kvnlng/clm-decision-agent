"""Tools for the data service desk demo. Each reaches a different kind of data source:

    lookup_queue    a local JSON file
    find_dataset    a SQLite database, through a parameterised query
    search_tickets, get_ticket, create_ticket    an issue tracker's REST API

The tracker calls follow the REST API of Forgejo and Gitea, which the mock tracker imitates.
The scenario's [tools.settings] table is passed to configure(): tracker_url, repository
("owner/name") and token_env, the name of the environment variable that holds an access token.
For another tracker, rewrite the three tracker tools; nothing else depends on them.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import httpx
from langchain_core.tools import ToolException, tool

SETTINGS: dict = {}
TICKET_NUMBER = re.compile(r"#?\d+")
PRIORITIES = ("low", "medium", "high")
PATH_SETTINGS = ("queues_file", "catalogue_db", "catalogue_sql")


def configure(settings: dict, scenario_dir: Path) -> None:
    """Take the scenario's settings; build the catalogue database from its SQL file if it is not there yet."""
    SETTINGS.clear()
    SETTINGS.update(settings)
    for key in PATH_SETTINGS:
        if key in SETTINGS:
            SETTINGS[key] = str(Path(scenario_dir) / SETTINGS[key])
    database, script = SETTINGS.get("catalogue_db"), SETTINGS.get("catalogue_sql")
    if database and script and not Path(database).exists():
        # Built under another name and renamed, so a failed build cannot leave a half-made catalogue
        statements, building = Path(script).read_text(encoding="utf-8"), Path(database + ".building")
        building.unlink(missing_ok=True)
        try:
            with closing(sqlite3.connect(building)) as connection:
                connection.executescript(statements)
                connection.commit()
            building.replace(database)
        finally:
            building.unlink(missing_ok=True)


@tool
def lookup_queue(request_type: str) -> dict:
    """Find which team handles a kind of request, the approvals it needs and the usual turnaround.
    request_type is one of: Data_Access, Research_Support, Information_Request."""
    queues = json.loads(Path(SETTINGS["queues_file"]).read_text(encoding="utf-8"))
    if request_type not in queues:
        raise ToolException(f"No queue for '{request_type}'. Known request types: {', '.join(queues)}")
    return queues[request_type]


@tool
def find_dataset(search_term: str) -> list[dict]:
    """Search the catalogue of datasets by name or description.
    Returns up to five datasets with name, description, steward and sensitivity."""
    words = [word for word in re.findall(r"\w+", search_term) if len(word) > 2][:5]
    if not words:
        return []
    # One pair of placeholders per word; the words themselves never become part of the SQL text
    condition = " OR ".join("name LIKE ? OR description LIKE ?" for _ in words)
    values = [f"%{word}%" for word in words for _ in range(2)]
    with closing(sqlite3.connect(SETTINGS["catalogue_db"])) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT name, description, steward, sensitivity FROM datasets WHERE {condition} ORDER BY name LIMIT 5",
            values).fetchall()
    return [dict(row) for row in rows]


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
        raise ToolException(
            f"The tracker at {base} could not be reached ({type(error).__name__}). "
            f"For the demo, start it with: uv run python examples/service_desk/mock_tracker.py") from error


def _checked(response: httpx.Response, action: str, expected: int = 200):
    """The response's JSON, or a ToolException that says what the tracker answered."""
    if response.status_code in (401, 403):
        raise ToolException(f"The tracker rejected the credentials (HTTP {response.status_code}). "
                            f"Check the token in the variable named by token_env.")
    if response.status_code != expected:
        try:
            reason = response.json().get("message") or f"HTTP {response.status_code}"
        except (ValueError, AttributeError):
            reason = f"HTTP {response.status_code}"
        raise ToolException(f"The tracker could not {action}: {reason}")
    return response.json()


def _summary(issue: dict) -> dict:
    return {"number": issue["number"], "summary": issue.get("title"), "status": issue.get("state"),
            "labels": [label["name"] for label in issue.get("labels") or []], "url": issue.get("html_url")}


@tool
def search_tickets(text: str) -> list[dict]:
    """Search existing tickets for ones about the same thing. Returns number, summary and status of up to five."""
    words = " ".join(re.findall(r"\w+", text))
    if not words:
        return []
    found = _checked(_request("GET", "/issues", params={"q": words, "state": "all", "type": "issues", "limit": 5}),
                     "search")
    return [{key: issue[key] for key in ("number", "summary", "status")} for issue in map(_summary, found)]


@tool
def get_ticket(number: str) -> dict:
    """Fetch one ticket by its number, for example 12. Returns number, summary, status, labels and url."""
    number = number.strip()
    if not TICKET_NUMBER.fullmatch(number):       # it becomes part of the address, so it must be only a number
        raise ToolException(f"'{number}' is not a ticket number. A ticket number looks like 12.")
    response = _request("GET", f"/issues/{number.lstrip('#')}")
    if response.status_code == 404:
        raise ToolException(f"No ticket with the number {number.lstrip('#')}.")
    return _summary(_checked(response, "fetch the ticket"))


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


@tool
def create_ticket(summary: str, description: str, priority: str = "Medium", labels: list[str] | None = None) -> dict:
    """Submit a new ticket to the tracker. summary is one line; description holds every collected detail.
    priority is Low, Medium or High. labels is a list of short tags. Returns the new ticket's number and url."""
    if not summary.strip():
        raise ToolException("A ticket needs a summary.")     # the tracker itself would accept a blank one
    if priority.strip().lower() not in PRIORITIES:
        raise ToolException("priority must be Low, Medium or High.")
    names = [f"priority-{priority.strip().lower()}"]
    for label in labels or []:
        name = "-".join(re.findall(r"\w+", label.lower()))
        if name and name not in names and not name.startswith("priority-"):
            names.append(name)
    created = _checked(_request("POST", "/issues", json={"title": summary.strip(), "body": description,
                                                         "labels": _label_ids(names)}), "create the ticket", 201)
    return {"number": created["number"], "url": created.get("html_url"), "status": "created"}


TOOLS = [lookup_queue, find_dataset, search_tickets, get_ticket, create_ticket]
