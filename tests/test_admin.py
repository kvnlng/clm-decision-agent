"""Tests for admin.py: tools the operator disables, hidden from the list and refused, and the trace's two new fields."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json
import stat

import anyio
import pytest
from conftest import ENV_TOML
from mcp import Client
from mcp.server.subscriptions import ToolsListChanged
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS

from clm_decision import server as server_module
from clm_decision import tracing
from clm_decision.admin import Admin
from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server, reload
from clm_decision.scenario import ScenarioError
from clm_decision.serving import Serve

NO_SCENARIO = ENV_TOML.split("[scenarios.sourcing]", maxsplit=1)[0]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def served(write_environment, tmp_path):
    """A server for the toy environment with an Admin whose file is in tmp_path."""
    admin = Admin(tmp_path / "disabled_tools.json")
    return build_environment_server(load_environment(write_environment()), admin=admin), admin


@pytest.mark.anyio
async def test_disabled_tool_hidden_and_refused(served):
    server, admin = served
    admin.disable("query_suppliers")
    async with Client(server) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        with pytest.raises(MCPError) as refused:
            await client.call_tool("query_suppliers", {"product": "P1"})
        admin.enable("query_suppliers")
        again = await client.call_tool("query_suppliers", {"product": "P1"})
    assert "query_suppliers" not in names and "describe" in names
    assert refused.value.code == INVALID_PARAMS and refused.value.message == "tool disabled by the operator"
    assert not again.is_error


def test_describe_cannot_be_disabled(served):
    _, admin = served
    with pytest.raises(ValueError, match="describe"):
        admin.disable("describe")
    with pytest.raises(ValueError, match="describe"):
        admin.enable("describe")


def test_an_unknown_name_is_refused_with_the_tools_there_are(served):
    _, admin = served
    with pytest.raises(ValueError, match="nope.*query_suppliers"):
        admin.disable("nope")
    with pytest.raises(ValueError, match="nope"):
        admin.enable("nope")


def test_disabled_persists(served, tmp_path):
    _, admin = served
    admin.disable("query_suppliers")
    admin.disable("calculation_cost")
    path = tmp_path / "disabled_tools.json"
    assert json.loads(path.read_text()) == {"disabled": ["calculation_cost", "query_suppliers"]}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert Admin(path).disabled == frozenset({"calculation_cost", "query_suppliers"})
    admin.enable("calculation_cost")
    assert Admin(path).disabled == frozenset({"query_suppliers"})


def test_a_missing_file_disables_nothing_and_a_bad_one_stops_start_up(tmp_path):
    assert Admin(tmp_path / "disabled_tools.json").disabled == frozenset()
    assert Admin.for_environment(tmp_path / "environment.toml").path == tmp_path / "disabled_tools.json"
    for bad in ("not json", '{"disabled": "query_x"}', '["query_x"]', '{"disabled": [1]}'):
        (tmp_path / "bad.json").write_text(bad)
        with pytest.raises(ScenarioError, match="bad.json"):
            Admin(tmp_path / "bad.json")


@pytest.mark.anyio
async def test_list_carries_ttl(served):
    server, _ = served
    async with Client(server) as client:
        listed = await client.list_tools()
    assert listed.ttl_ms == 30000


@pytest.mark.anyio
async def test_describe_lists_disabled(served):
    server, admin = served
    admin.disable("query_suppliers")
    async with Client(server) as client:
        described = (await client.call_tool("describe", {})).structured_content
    assert described["disabled"] == ["query_suppliers"] and "suppliers" not in described["queries"]
    admin.enable("query_suppliers")
    async with Client(server) as client:
        described = (await client.call_tool("describe", {})).structured_content
    assert "disabled" not in described and "suppliers" in described["queries"]


@pytest.mark.anyio
async def test_a_name_disabled_in_the_file_hides_only_an_offered_tool(write_environment, tmp_path):
    path = tmp_path / "disabled_tools.json"
    written = json.dumps({"disabled": ["query_suppliers", "scenario_sourcing", "gone_tool"]})
    path.write_text(written)
    server = build_environment_server(load_environment(write_environment()), admin=Admin(path),
                                      serve=Serve(tools=("describe", "query_suppliers", "calculation_cost")))
    async with Client(server) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        described = (await client.call_tool("describe", {})).structured_content
    assert names == {"describe", "calculation_cost"}
    assert described["disabled"] == ["query_suppliers"]
    assert path.read_text() == written


@pytest.mark.anyio
async def test_changed_and_a_reload_publish_tools_list_changed(write_environment, tmp_path):
    admin = Admin(tmp_path / "disabled_tools.json")
    server = build_environment_server(load_environment(write_environment()), admin=admin)
    heard = []
    admin.bus.subscribe(heard.append)
    await admin.changed()
    reload(server, load_environment(write_environment(NO_SCENARIO, folder="new")))
    for _ in range(10):     # the reload's publish is scheduled on the loop
        await anyio.sleep(0)
    assert [type(event) for event in heard] == [ToolsListChanged, ToolsListChanged]


@pytest.mark.anyio
async def test_trace_has_size_and_key_id(write_environment, tmp_path):
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    server = build_environment_server(load_environment(write_environment()), trace=writer)
    token = tracing.CALLER_KEY_ID.set("0123abcd")
    try:
        async with Client(server) as client:
            result = await client.call_tool("query_suppliers", {"product": "P1"})
    finally:
        tracing.CALLER_KEY_ID.reset(token)
    writer.close()
    record = json.loads((tmp_path / "trace.jsonl").read_text())
    assert record["caller"] == {"key_id": "0123abcd"}
    assert record["result_size_bytes"] == len(json.dumps(result.structured_content).encode())


def test_clm_server_applies_the_file_beside_the_environment_and_stops_on_a_bad_one(write_environment, monkeypatch,
                                                                                   capsys):
    toml = write_environment()
    built = {}
    monkeypatch.setattr(server_module, "build_environment_server",
                        lambda *args, **kwargs: built.update(kwargs) or type("Ran", (), {"run": lambda *a: None})())
    (toml.parent / "disabled_tools.json").write_text('{"disabled": ["query_suppliers"]}')
    assert server_module.main([str(toml)]) == 0
    assert built["admin"].disabled == frozenset({"query_suppliers"})
    (toml.parent / "disabled_tools.json").write_text("{")
    assert server_module.main([str(toml)]) == 2
    error = capsys.readouterr().err.strip()
    assert error.startswith("error: cannot read the disabled tools in ") and "disabled_tools.json" in error
    assert len(error.splitlines()) == 1
