"""Tests for call.py: clm-call lists a server's tools or calls one, over stdio or HTTP."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json
from pathlib import Path

import pytest
from conftest import free_port, serving

from clm_decision import server
from clm_decision.call import main
from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server

EXAMPLE = Path(__file__).parent.parent / "examples" / "tariff" / "environment.toml"


def test_without_a_tool_it_lists_the_tools(capsys):
    assert main([str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "scenario_sourcing(product, branches)" in out and "Where to source" in out and "describe()" in out


def test_a_call_prints_the_result_as_json(capsys):
    assert main([str(EXAMPLE), "scenario_sourcing", '{"product": "PX"}']) == 0
    assert json.loads(capsys.readouterr().out)["best"] == {"region_b_share": 0.5}


def test_a_tool_error_is_one_line(capsys):
    assert main([str(EXAMPLE), "scenario_sourcing", '{"product": "PX", "branches": [{"region_b_share": 0.7}]}']) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "region_b_share = 0.7 is not allowed" in err and "Traceback" not in err


@pytest.mark.parametrize("arguments", ["{product: PX}", '["PX"]'])
def test_arguments_must_be_a_json_object(arguments, capsys):
    assert main([str(EXAMPLE), "scenario_sourcing", arguments]) == 2
    assert capsys.readouterr().err.startswith("error: the arguments must be a JSON object")


def test_a_server_that_stops_is_one_line(tmp_path, capsys):
    assert main([str(tmp_path / "nowhere.toml")]) == 2
    assert "error: the decision server stopped" in capsys.readouterr().err


@pytest.fixture
def url():
    port = free_port()
    with serving(server.http_app(build_environment_server(load_environment(EXAMPLE)), "k1"), port):
        yield f"http://127.0.0.1:{port}/mcp"


def test_over_http_with_the_key(url, monkeypatch, capsys):
    monkeypatch.setenv("DECISION_SERVER_KEY", "k1")
    assert main([url, "describe"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "Component sourcing"


def test_over_http_with_a_wrong_key_is_one_line(url, monkeypatch, capsys):
    monkeypatch.setenv("DECISION_SERVER_KEY", "wrong")
    assert main([url]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"error: cannot connect to {url}") and "401" in err
