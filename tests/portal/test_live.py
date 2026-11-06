"""Tests for the server agents use as the portal sees it in the same process: reloads, testing and the two ports.

They run a real LiveServer over a real environment with a derived layer, its documents in memory (fake_documents),
so no test needs PostgreSQL, a model or a second process.
"""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import asyncio
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
from fake_documents import FakeEmbedder
from mcp import Client
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from test_derived_load import PARTS, entry, make, store

from clm_decision.admin import Admin
from clm_decision.checks import Checks
from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server
from clm_decision.portal import live, managed, web
from clm_decision.scenario import ScenarioError
from .conftest import KEY, ORIGIN, post, sign_in
from .test_chat import reply, scripted


def build(path: Path) -> live.LiveServer:
    """The agents' server on the environment at `path`, its documents in memory, as clm-server --portal builds it."""
    admin = Admin.for_environment(path)
    server = build_environment_server(load_environment(path, store=store(), embedder=FakeEmbedder()), admin=admin)
    return live.LiveServer(server, admin, Checks(server, None, None))


@pytest.fixture
def derived_environment(tmp_path) -> Path:
    """An environment whose derived decision `carry` has passed its check and is not yet approved."""
    return make(tmp_path, decision=entry(status="checked"))


@pytest.fixture
def server(derived_environment) -> live.LiveServer:
    return build(derived_environment)


@pytest.fixture
def client(derived_environment, server, monkeypatch):
    monkeypatch.setattr(web, "WRONG_KEY_WAIT", 0.01)
    portal = web.Portal(derived_environment, server, managed.Jobs(script=lambda name: None))
    with TestClient(web.create_app(portal, KEY, "127.0.0.1", 8400), base_url=ORIGIN) as test_client:
        yield test_client


def agents_tools(server: live.LiveServer) -> list[str]:
    """The tools the agents' server lists, through a client of its own."""
    async def listed():
        async with Client(server.server) as found:
            return [tool.name for tool in (await found.list_tools()).tools]
    return asyncio.run(listed())


def agents_call(server: live.LiveServer, name: str, arguments: dict) -> dict:
    async def called():
        async with Client(server.server) as found:
            return live.result_dict(name, await found.call_tool(name, arguments))
    return asyncio.run(called())


def test_approve_reloads_not_restarts(client, server, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("a process was started"))
    agents = server.server
    assert "decision_carry" not in agents_tools(server)
    token = sign_in(client)
    response = post(client, "/review/decisions/carry/approve", token)
    assert response.status_code == 200 and "carry approved." in response.text, response.text
    assert server.server is agents and server.error is None
    assert agents_call(server, "decision_carry", {"years": 3})["outputs"]["carry"] == 7


def test_failed_reload_keeps_serving(client, server, derived_environment):
    token = sign_in(client)
    derived_environment.write_text(derived_environment.read_text() + "\n[queries.broken\n")
    response = post(client, "/reload", token)
    assert response.status_code == 409 and "the last reload failed" in response.text
    assert "Load errors" in client.get("/").text and "the last reload failed" in client.get("/").text
    assert agents_call(server, "decision_other", {"x": 2})["outputs"] == {"y": 2}
    derived_environment.write_text(derived_environment.read_text().replace("\n[queries.broken\n", ""))
    assert post(client, "/reload", token, follow_redirects=False).status_code == 303 and server.error is None


def test_testing_is_private(client, server):
    seen: list[dict] = []
    portal = client.app.app.state.portal

    def asked() -> list[str]:
        return [tool["function"]["name"] for tool in seen[-1]["tools"]]
    portal.make_model = lambda *args: scripted([reply("Answered.")], seen)
    token = sign_in(client)
    post(client, "/ask", token, {"question": "Which rules?", "backend": "openai"})
    assert "decision_carry" not in asked()
    assert post(client, "/ask/testing", token, {"on": "1"}, follow_redirects=False).status_code == 303
    assert server.testing and web.TESTING_BANNER in client.get("/").text
    post(client, "/ask", token, {"question": "And now?", "backend": "openai"})
    assert "decision_carry" in asked() and "decision_carry" not in agents_tools(server)
    assert post(client, "/reload", token, follow_redirects=False).status_code == 303
    post(client, "/ask", token, {"question": "After a reload?", "backend": "openai"})
    assert server.testing and "decision_carry" in asked() and "decision_carry" not in agents_tools(server)
    post(client, "/ask/testing", token, {"on": "0"})
    post(client, "/ask", token, {"question": "Off?", "backend": "openai"})
    assert not server.testing and "decision_carry" not in asked()


def test_testing_that_cannot_start_stays_off(tmp_path):
    server = build(make(tmp_path, decision=entry(status="checked"), slots="",
                        parts=PARTS.replace('limits = ["carry"]', "")))
    with pytest.raises(ScenarioError, match="no \\[derived\\] table"):
        asyncio.run(server.set_testing(True))
    assert not server.testing


def test_a_failed_reload_never_shows_a_secret(server, monkeypatch):
    monkeypatch.setenv("CLM_DOCUMENTS_PASSWORD", "a-long-database-password")

    def refuse(*args, **kwargs):
        raise ScenarioError("cannot connect as clm:a-long-database-password@localhost")
    monkeypatch.setattr(live, "load_environment", refuse)
    asyncio.run(server.reload())
    assert "a-long-database-password" not in server.error and "clm:***@localhost" in server.error


def test_the_checks_run_while_the_portal_runs(derived_environment, server, monkeypatch):
    monkeypatch.setattr(web, "CHECK_SECONDS", 0.01)
    portal = web.Portal(derived_environment, server, managed.Jobs(script=lambda name: None))
    with TestClient(web.create_app(portal, KEY, "127.0.0.1", 8400), base_url=ORIGIN):
        for _ in range(500):
            if len(server.checks.latest) == 7:
                break
            time.sleep(0.01)
    assert [check.name for check in server.checks.latest][-2:] == ["tools", "describe"]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def answers(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def test_both_ports_stop():
    app = Starlette(routes=[Route("/", lambda request: PlainTextResponse("up"))])
    ports = free_port(), free_port()
    stops, removed = [], []

    def on_signal(stop):
        stops.append(stop)
        return lambda: removed.append(True)
    thread = threading.Thread(target=live.serve_both, args=(app, app, ("127.0.0.1", ports[0]),
                                                            ("127.0.0.1", ports[1]), on_signal))
    thread.start()
    try:
        for _ in range(200):
            if all(answers(port) for port in ports):
                break
            time.sleep(0.02)
        assert all(answers(port) for port in ports) and len(stops) == 1
    finally:
        if stops:
            stops[0]()      # as SIGINT or SIGTERM would
        thread.join(10)
    assert not thread.is_alive() and removed == [True]
    assert not any(answers(port) for port in ports)
