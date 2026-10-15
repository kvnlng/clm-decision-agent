"""Tools for the data service desk demo. Each reaches a different kind of data source:

    lookup_queue    a local JSON file
    list_datasets, find_dataset    a data catalogue's REST API (OpenMetadata's), read once
    search_tickets, get_ticket, create_ticket    an issue tracker's REST API

list_datasets is also the scenario's catalogue tool: the framework calls it at start-up and CLM
chooses between the datasets it returns (see [catalogue] in scenario.toml).

The tracker calls follow the REST API of Forgejo and Gitea, which the mock tracker imitates.
The scenario's [tools.settings] table is passed to configure(): tracker_url, repository
("owner/name") and token_env, the name of the environment variable that holds an access token;
and openmetadata_url, openmetadata_token_env and, optionally, openmetadata_domain.
For another tracker, rewrite the three tracker tools; nothing else depends on them.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
import openmetadata_catalogue
from langchain_core.tools import ToolException, tool

SETTINGS: dict = {}
TICKET_NUMBER = re.compile(r"#?\d+")
PRIORITIES = ("low", "medium", "high")
PATH_SETTINGS = ("queues_file",)
_DATASETS: list = []        # the catalogue, once read


def configure(settings: dict, scenario_dir: Path) -> None:
    """Take the scenario's settings. The catalogue is read again on first use."""
    SETTINGS.clear()
    SETTINGS.update(settings)
    for key in PATH_SETTINGS:
        if key in SETTINGS:
            SETTINGS[key] = str(Path(scenario_dir) / SETTINGS[key])
    _DATASETS.clear()


@tool
def lookup_queue(request_type: str) -> dict:
    """Find which team handles a kind of request, the approvals it needs and the usual turnaround.
    request_type is one of: Data_Access, Research_Support, Information_Request."""
    queues = json.loads(Path(SETTINGS["queues_file"]).read_text(encoding="utf-8"))
    if request_type not in queues:
        raise ToolException(f"No queue for '{request_type}'. Known request types: {', '.join(queues)}")
    return queues[request_type]


def _datasets() -> list[dict]:
    """The catalogue, read from OpenMetadata on first use and kept."""
    if not _DATASETS:
        token = os.environ.get(SETTINGS.get("openmetadata_token_env", ""), "") or None
        tables, lineage = openmetadata_catalogue.fetch(SETTINGS["openmetadata_url"], token,
                                                       SETTINGS.get("openmetadata_domain"))
        _DATASETS.extend(openmetadata_catalogue.to_entries(tables, lineage))
    return _DATASETS


@tool
def list_datasets() -> list[dict]:
    """List every dataset in the catalogue, with its description and what is known about it:
    who owns it, how sensitive it is and what it is built from."""
    return _datasets()


@tool
def find_dataset(search_term: str) -> list[dict]:
    """Search the catalogue of datasets by name or description.
    Returns up to five datasets with name, description and what is known about each."""
    words = [word.lower() for word in re.findall(r"\w+", search_term) if len(word) > 2][:5]
    found = [entry for entry in _datasets()
             if any(word in f"{entry['name']} {entry['description']}".lower() for word in words)]
    return sorted(found, key=lambda entry: entry["name"])[:5]


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


TOOLS = [lookup_queue, find_dataset, list_datasets, search_tickets, get_ticket, create_ticket]
