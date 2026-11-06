"""Tests for the Server page: state, checks, tool switches and credentials, over a real LiveServer, Admin and Checks.

No credential may appear on the page; the keys are known by their 8-hex ids and the passwords only as set or missing.
"""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import hashlib
from pathlib import Path

import pytest
from starlette.testclient import TestClient
from test_derived_load import entry, make

from clm_decision.documents import DocumentSettings
from clm_decision.portal import managed, web
from clm_decision.server import KeySet
from .conftest import KEY, ORIGIN, post, sign_in
from .test_live import agents_tools, build

AGENT_KEY = "agent-key-that-is-long-enough-1"
OTHER_KEY = "second-agent-key-long-enough-2"
PASSWORD = "a-long-database-password"


def ident(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:8]


@pytest.fixture
def environment_file(tmp_path) -> Path:
    return make(tmp_path, decision=entry(status="approved"))


@pytest.fixture
def live_server(environment_file):
    return build(environment_file)


@pytest.fixture
def client(environment_file, live_server, monkeypatch):
    monkeypatch.setattr(web, "WRONG_KEY_WAIT", 0.01)
    monkeypatch.setenv("CLM_DOCUMENTS_PASSWORD", PASSWORD)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    env = live_server.server.holder.current
    settings = DocumentSettings("localhost", 5432, "clm", "clm", "CLM_DOCUMENTS_PASSWORD", "http://localhost:11434")
    object.__setattr__(env, "document_settings", settings)      # the tiny environment has no PostgreSQL
    keys = KeySet.from_sources(AGENT_KEY, None)
    keys.match(AGENT_KEY.encode())
    portal = web.Portal(environment_file, live_server, managed.Jobs(script=lambda name: None), keys=keys)
    with TestClient(web.create_app(portal, KEY, "127.0.0.1", 8400), base_url=ORIGIN) as test_client:
        yield test_client


def test_server_page_shows_state_checks_tools_credentials(client, live_server):
    token = sign_in(client)
    assert post(client, "/server/checks", token, follow_redirects=False).status_code == 303
    text = client.get("/server").text
    for shown in ("Environment file", "MCP protocol version", "Started", "Authentication", "Tool list fingerprint",
                  "Run checks now", "describe answered", "decision_carry", "enabled", "Credentials",
                  f"Agents' key {ident(AGENT_KEY)}", "LLM_API_KEY"):
        assert shown in text, shown
    assert live_server.server.holder.current.name in text and live_server.checks.baseline[:16] in text
    assert '<a href="/server">Server</a>' in text
    assert "/server/tools/describe/" not in text and "/server/tools/decision_carry/disable" in text


def test_disable_from_page(client, live_server):
    token = sign_in(client)
    assert "decision_carry" in agents_tools(live_server)
    assert post(client, "/server/tools/decision_carry/disable", token, follow_redirects=False).status_code == 303
    assert "decision_carry" not in agents_tools(live_server)
    assert "/server/tools/decision_carry/enable" in client.get("/server").text
    assert post(client, "/server/tools/decision_carry/enable", token, follow_redirects=False).status_code == 303
    assert "decision_carry" in agents_tools(live_server)


def test_a_refused_switch_shows_its_message(client):
    token = sign_in(client)
    response = post(client, "/server/tools/describe/disable", token)
    assert response.status_code == 400 and "describe cannot be disabled" in response.text
    response = post(client, "/server/tools/nonesuch/disable", token)
    assert response.status_code == 400 and "no tool &#x27;nonesuch&#x27;" in response.text


def test_a_switch_that_is_neither_disable_nor_enable_is_not_found(client, live_server):
    token = sign_in(client)
    post(client, "/server/tools/decision_carry/disable", token)
    response = post(client, "/server/tools/decision_carry/toggle", token)
    assert response.status_code == 404 and "decision_carry" not in agents_tools(live_server)


def test_no_secret_on_page(client):
    token = sign_in(client)
    post(client, "/server/checks", token)
    text = client.get("/server").text
    for secret in (KEY, AGENT_KEY, PASSWORD):
        assert secret not in text and hashlib.sha256(secret.encode()).hexdigest() not in text
    assert ident(AGENT_KEY) in text
    assert "PostgreSQL password (CLM_DOCUMENTS_PASSWORD)</td><td>set<" in text
    assert "<td>Portal key</td><td>set<" in text


def test_a_second_key_is_never_used(client):
    keys = client.app.app.state.portal.keys
    keys._digests.append(hashlib.sha256(OTHER_KEY.encode()).hexdigest())  # pylint: disable=protected-access
    sign_in(client)
    text = client.get("/server").text
    assert f"{ident(OTHER_KEY)}</td><td>last used never" in text and OTHER_KEY not in text


def test_missing_password_and_oauth_settings(client, monkeypatch):
    from clm_decision import oauth  # pylint: disable=import-outside-toplevel
    monkeypatch.delenv("CLM_DOCUMENTS_PASSWORD")
    portal = client.app.app.state.portal
    portal.keys, portal.oauth = None, oauth.oauth("https://issuer.example/realm", "api://audience",
                                                    ["decide", "read"], "http://127.0.0.1:8000/mcp")
    sign_in(client)
    text = client.get("/server").text
    assert "CLM_DOCUMENTS_PASSWORD)</td><td>missing<" in text and "OAuth" in text
    for shown in ("https://issuer.example/realm", "api://audience", "decide read"):
        assert shown in text


def test_page_says_stale_list_window(client):
    sign_in(client)
    text = client.get("/server").text
    assert "up to 30 seconds" in text and "older client" in text


def test_forms_need_csrf(client, live_server):
    sign_in(client)
    for path in ("/server/checks", "/server/tools/decision_carry/disable"):
        response = client.post(path, data={}, headers={"Origin": ORIGIN})
        assert response.status_code == 403, path
        response = client.post(path, data={"csrf": "wrong"}, headers={"Origin": ORIGIN})
        assert response.status_code == 403, path
    assert "decision_carry" in agents_tools(live_server)
    assert TestClient(client.app, base_url=ORIGIN).get("/server", follow_redirects=False).status_code == 303
