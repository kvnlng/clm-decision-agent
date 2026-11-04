"""Tests for connection.py: how clm-agent reaches a decision server, a file over stdio or a URL over HTTP."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from pathlib import Path

import anyio
import pytest
from conftest import free_port, serving
from mcp import Client

from clm_decision import server
from clm_decision.connection import connect, is_url, refusal
from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server

EXAMPLE = Path(__file__).parent.parent / "examples" / "tariff" / "environment.toml"


async def _tools(target: str, refused: list[int] | None = None) -> set[str]:
    async with Client(connect(target, refused)) as client:
        return {tool.name for tool in (await client.list_tools()).tools}


def test_a_file_is_started_over_stdio():
    assert not is_url(str(EXAMPLE))
    assert "scenario_sourcing" in anyio.run(_tools, str(EXAMPLE))


@pytest.fixture
def url():
    port = free_port()
    with serving(server.http_app(build_environment_server(load_environment(EXAMPLE)), "k1"), port):
        yield f"http://127.0.0.1:{port}/mcp"


def test_a_url_is_reached_over_http_with_the_key(url, monkeypatch):
    monkeypatch.setenv("DECISION_SERVER_KEY", "k1")
    assert is_url(url) and "describe" in anyio.run(_tools, url)


def test_a_wrong_key_is_noted_as_a_refusal(url, monkeypatch):
    monkeypatch.setenv("DECISION_SERVER_KEY", "wrong")
    refused: list[int] = []
    with pytest.raises(BaseException):
        anyio.run(_tools, url, refused)
    assert refusal(refused) == "the server refused the call (401), with the key in DECISION_SERVER_KEY"
    assert refusal([]) is None
