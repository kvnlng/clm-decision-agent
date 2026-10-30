"""Tests for environment_server.py, and server.py serving an environment: one MCP tool per named part, no CLM."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json
from pathlib import Path

import pytest
from conftest import ENV_TOML, JUDGE_TOML, FakeRouter, decision
from mcp import Client

from clm_decision import clm_router
from clm_decision import server
from clm_decision import tracing
from clm_decision.environment import Environment, load_environment
from clm_decision.environment_server import build_environment_server

EXAMPLE = Path(__file__).parent.parent / "examples" / "tariff" / "environment.toml"
QUERIES = ["product_facts", "components_of", "suppliers_of", "key_clients_of", "supply_dependencies",
           "single_points_of_failure", "supplier_news"]
DECISIONS = ["tariff_rate", "sla_threshold", "quality_threshold", "supply_concentration", "news_risk"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def mcp_server():
    return build_environment_server(load_environment(EXAMPLE))


@pytest.mark.anyio
async def test_one_tool_per_part(mcp_server):
    async with Client(mcp_server) as client:
        names = sorted(tool.name for tool in (await client.list_tools()).tools)
    assert names == sorted([f"query_{name}" for name in QUERIES] + [f"decision_{name}" for name in DECISIONS]
                           + ["calculation_branch_figures", "calculation_concentration", "scenario_sourcing",
                              "scenario_sourcing_with_news", "judge_delivery_risk", "describe"])


@pytest.mark.anyio
async def test_a_tool_schema_has_the_declared_params(mcp_server):
    async with Client(mcp_server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    schema = tools["scenario_sourcing"].input_schema
    assert schema["required"] == ["product"] and schema["properties"]["product"]["type"] == "string"
    assert "branches" in schema["properties"]
    assert tools["decision_sla_threshold"].input_schema["properties"]["extension_days"]["type"] == "number"
    assert tools["query_supply_dependencies"].input_schema["properties"]["suppliers"]["type"] == "array"
    assert tools["scenario_sourcing"].description.startswith("Where to source")
    assert tools["scenario_sourcing"].annotations.read_only_hint


@pytest.mark.anyio
async def test_scenario_over_mcp(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool("scenario_sourcing", {"product": "PX"})
        chosen = await client.call_tool("scenario_sourcing", {"product": "PX", "branches": [{"region_b_share": 0}]})
        described = await client.call_tool("describe", {})
    assert not result.is_error and result.structured_content["best"] == {"region_b_share": 0.5}
    assert [branch["levers"] for branch in chosen.structured_content["branches"]] == [{"region_b_share": 0}]
    sourcing = described.structured_content["scenarios"]["sourcing"]
    assert sourcing["levers"] == {"region_b_share": [0, 0.5, 1.0]} and sourcing["objective"] == "q3_margin"
    assert described.structured_content["name"] == "Component sourcing"


@pytest.mark.anyio
async def test_a_bad_call_is_one_line(mcp_server):
    async with Client(mcp_server) as client:
        missing = await client.call_tool("query_suppliers_of", {})
        lever = await client.call_tool("scenario_sourcing", {"product": "PX", "branches": [{"region_b_share": 0.7}]})
        unknown = await client.call_tool("query_suppliers_of", {"component": "C9"})
    assert missing.is_error and "component" in missing.content[0].text
    assert lever.is_error and lever.content[0].text.count("\n") == 0
    assert "region_b_share = 0.7 is not allowed" in lever.content[0].text
    assert not unknown.is_error and unknown.structured_content["rows"] == []


def test_server_main_serves_an_environment_without_clm(monkeypatch):
    def no_clm(scenario):
        raise AssertionError("CLM was loaded")
    monkeypatch.setattr(clm_router, "make_router", no_clm)
    assert isinstance(server.load_any(str(EXAMPLE)), Environment)


def test_a_scenario_file_still_loads_clm(write_scenario, monkeypatch):
    router = FakeRouter(decision("Tech"))
    router.load_verdicts = lambda records, scenario: False      # no corrections recorded yet
    monkeypatch.setattr(clm_router, "make_router", lambda scenario: (router, "test heads"))
    scenario, loaded, tools = server.load_any(str(write_scenario()))
    assert scenario.path.name == "scenario.toml" and loaded is router and "lookup" in tools


def test_review_is_refused_for_an_environment(capsys):
    assert server.main([str(EXAMPLE), "--review"]) == 2
    assert "review mode is for CLM scenarios" in capsys.readouterr().err


@pytest.mark.anyio
async def test_a_traced_call_writes_its_tool_and_outcome(write_environment, tmp_path):
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    traced = build_environment_server(load_environment(write_environment()), trace=writer)
    async with Client(traced) as client:
        await client.call_tool("query_suppliers", {"product": "P1"})
        failed = await client.call_tool("scenario_sourcing", {"product": "P1", "branches": [{"share": 0.7}]})
    writer.close()
    done, undone = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert failed.is_error
    assert (done["tool"], done["outcome"]) == ("query_suppliers", {"ok": True}) and done["ms"] >= 0
    assert (undone["tool"], undone["outcome"]) == ("scenario_sourcing", {"error": "ToolError"})


@pytest.mark.anyio
async def test_an_owner_error_reaches_the_agent(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool("query_product_facts", {"product": "nope"})
    assert result.is_error and "KeyError: 'nope'" in result.content[0].text


@pytest.mark.anyio
async def test_judge_tool_over_mcp(write_environment, fake_clm):  # pylint: disable=unused-argument
    env = load_environment(write_environment(ENV_TOML + JUDGE_TOML))
    async with Client(build_environment_server(env)) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        judged = await client.call_tool("judge_risk", {"text": "a strike is called"})
        described = await client.call_tool("describe", {})
    assert "judge_risk" in names and judged.structured_content["result"] == {"probability": 0.9}
    assert described.structured_content["judges"]["risk"]["instructions"].startswith("Does this news")
