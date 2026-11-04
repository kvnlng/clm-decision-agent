"""A tiny environment on disk, a stand-in for the portal's server, and a signed-in test client of the page.

The environment has a built-in graph, one supplied DMN decision citing a document, and a derived layer with one
checked decision, one fact whose check failed and one open conflict. The stand-in server answers `describe` and
`document_handbook` in memory, so no test needs a network, a model or PostgreSQL.
"""
# Fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import Client
from mcp.server import MCPServer
from starlette.testclient import TestClient

from clm_portal import managed, web

KEY = "test-portal-key-not-a-secret"
SERVER_KEY = "test-server-key-not-a-secret"
ORIGIN = "http://127.0.0.1:8400"

ENVIRONMENT = """\
[environment]
name = "Tiny handbook"
description = "A made-up handbook, for the portal's tests."
functions = "tiny_functions"

[graph]
backend = "builtin"
nodes = "graph/nodes.csv"
edges = "graph/edges.csv"

[documents]
backend = "postgres"

[collections.handbook]
description = "The made-up handbook."
embedding_model = "embeddinggemma"
filters = { policy = "string" }

[information.years_of_service]
description = "Full years of service."
kind = "supplied"

[decisions.vesting]
description = "How much of the match is the employee's."
dmn = "rules/vesting.dmn"
collection = "handbook"

[decisions.vesting.rule]
statement = "The match vests 25% for each full year."
reason = "The plan's vesting schedule."
owner = "People Operations (made up)"
version = "2026-01"
governs = ["benefits"]
needs = ["years_of_service"]
cites = ["hb-401k"]
"""


def dmn(name: str, output: str, rows: list[tuple[str, str]]) -> str:
    cells = "".join(f'<rule id="r{index}"><inputEntry><text>{when}</text></inputEntry>'
                    f'<outputEntry><text>{then}</text></outputEntry></rule>' for index, (when, then) in enumerate(rows))
    return (f'<?xml version="1.0" encoding="UTF-8"?><definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" '
            f'id="d" name="d" namespace="https://example.org/t"><decision id="{name}" name="{name}">'
            f'<decisionTable id="t" hitPolicy="FIRST"><input id="i"><inputExpression typeRef="number">'
            f'<text>years_of_service</text></inputExpression></input><output id="o" name="{output}" '
            f'typeRef="number"/>{cells}</decisionTable></decision></definitions>')


def source(document: str, passage: str, quote: str) -> dict:
    return {"document": document, "passages": [passage], "quote": quote, "edition": "2026-01-01"}


LAYER = {
    "version": 1, "collection": "handbook", "derived_at": "2026-11-04T10:00:00+00:00", "model": "stand-in",
    "decisions": {"carryover_limit": {
        "description": "How many unused days carry over.", "dmn": "rules/carryover_limit.dmn",
        "collection": "handbook",
        "rule": {"statement": "Up to 7 unused days carry over.", "reason": "The leave policy.",
                 "owner": "People Operations (made up)", "version": "2026-01", "governs": ["benefits"],
                 "cites": ["hb-leave"]},
        "source": source("hb-leave", "(2)", "up to 7 unused days carry into the next year"),
        "status": "checked", "check": {"passed": True, "failures": []}, "confidence": "high-ish",
        "decided_by": None, "decided_at": None, "note": None}},
    "facts": {"k401_2026.match_cap": {
        "node": "k401_2026", "property": "match_cap", "value": 5000,
        "source": source("hb-401k", "(4)", "the match is capped at $5,000"),
        "status": "proposed", "check": {"passed": False, "failures": ["the quote does not state 5000"]},
        "decided_by": None, "decided_at": None, "note": None}},
    "conflicts": {"k401_2026.match_cap": {
        "target": "facts", "name": "k401_2026.match_cap", "supplied": 6300, "derived": 5000,
        "source": source("hb-401k", "(4)", "the match is capped at $5,000"), "status": "open",
        "decided_by": None, "decided_at": None, "note": None}},
}


@pytest.fixture
def environment(tmp_path) -> Path:
    folder = tmp_path / "tiny"
    (folder / "rules").mkdir(parents=True)
    (folder / "graph").mkdir()
    (folder / "derived" / "rules").mkdir(parents=True)
    (folder / "environment.toml").write_text(ENVIRONMENT)
    (folder / "tiny_functions.py").write_text('"""No functions."""\n')
    (folder / "rules" / "vesting.dmn").write_text(dmn("vesting", "vested_percent", [("&gt;= 4", "100"), ("-", "0")]))
    (folder / "graph" / "nodes.csv").write_text("id,kind,name,match_cap\nmeridian,company,Meridian,\n"
                                                "k401_2026,plan,401(k) plan,6300\n")
    (folder / "graph" / "edges.csv").write_text("source,relation,target\nk401_2026,offered_by,meridian\n")
    (folder / "derived" / "rules" / "carryover_limit.dmn").write_text(
        dmn("carryover_limit", "limit", [("&gt;= 5", "7"), ("-", "5")]))
    (folder / "derived" / "derived.json").write_text(json.dumps(LAYER, indent=2))
    return folder / "environment.toml"


def stand_in() -> MCPServer:
    server = MCPServer("stand-in")

    @server.tool()
    def describe() -> dict:
        """What the environment offers."""
        return {"name": "Tiny handbook", "collections": {"handbook": {
            "description": "The made-up handbook.", "filters": {"policy": "string"},
            "built": {"built_at": "2026-11-01T09:30:00+00:00", "source": "handbook.jsonl", "documents": 2}}}}

    @server.tool()
    def document_handbook(id: str, passage: str | None = None) -> dict:  # pylint: disable=redefined-builtin
        """One document of handbook."""
        label = passage or "(1)"
        return {"document": {"id": id, "citation": f"Handbook {id}", "title": f"Policy {id}",
                             "passages": [{"label": label, "text": f"The passage {label} of {id}, as stored."}]}}

    @server.tool()
    def decision_vesting(years_of_service: float) -> dict:
        """How much of the match is the employee's."""
        return {"vested_percent": 100 if years_of_service >= 4 else 0}
    return server


class FakeServer:
    """The portal's server, in memory: it starts, restarts and fails as told, and serves the stand-in."""

    def __init__(self, error: str | None = None):
        self.server = stand_in()
        self.testing, self.error, self.failing = False, None, error
        self.starts: list[bool] = []
        self.url, self._key = "http://127.0.0.1:59999/mcp", SERVER_KEY

    @property
    def running(self) -> bool:
        return self.error is None and bool(self.starts)

    async def start(self, testing: bool = False) -> None:
        self.testing, self.error = testing, self.failing
        self.starts.append(testing)

    async def restart(self, testing: bool | None = None) -> None:
        await self.start(self.testing if testing is None else testing)

    def stop(self) -> None:
        pass

    def client(self) -> Client:
        return Client(self.server)

    async def tools(self) -> list:
        async with self.client() as client:
            return (await client.list_tools()).tools

    async def call(self, name: str, arguments: dict) -> dict:
        async with self.client() as client:
            return managed.result_dict(name, await client.call_tool(name, arguments))


def recorder(code: int = 0, output: str = "done\n"):
    """A job runner that records each command and, once `release` is cleared, waits for it to be set."""
    seen: list[list[str]] = []
    release = threading.Event()
    release.set()

    def run(argv, **_):
        seen.append(list(argv))
        release.wait(5)
        return SimpleNamespace(returncode=code, stdout=output, stderr="")
    run.seen, run.release = seen, release
    return run


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def runner():
    return recorder()


@pytest.fixture
def installed() -> set:
    """The console scripts the jobs find; take one out to see it missing."""
    return {"clm-server", "clm-load-documents", "clm-derive"}


@pytest.fixture
def portal(environment, server, runner, installed) -> web.Portal:
    jobs = managed.Jobs(runner=runner, script=lambda name: Path("/stand-in/bin") / name if name in installed else None)
    return web.Portal(environment, server, jobs, find=jobs.find)


@pytest.fixture
def client(portal, monkeypatch):
    monkeypatch.setattr(web, "WRONG_KEY_WAIT", 0.01)
    with TestClient(web.create_app(portal, KEY, "127.0.0.1", 8400), base_url=ORIGIN) as test_client:
        yield test_client


def sign_in(client, name: str = "Kevin") -> str:
    """Sign in and return the session's CSRF token, as the page's forms carry it."""
    response = client.post("/sign-in", data={"key": KEY, "name": name}, headers={"Origin": ORIGIN})
    assert response.status_code == 200, response.text
    return csrf_of(response.text)


def csrf_of(page: str) -> str:
    return re.search(r'name="csrf" value="([^"]+)"', page).group(1)


def post(client, path: str, token: str, data: dict | None = None, **kwargs):
    """A POST from the portal's own page."""
    return client.post(path, data={"csrf": token, **(data or {})}, headers={"Origin": ORIGIN}, **kwargs)
