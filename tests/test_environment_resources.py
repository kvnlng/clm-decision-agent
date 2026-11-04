"""Tests for the environment server's resources and prompt: the same data as the read tools, through a second door,
with a stand-in store so no PostgreSQL or Ollama is needed."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json
from pathlib import Path

import pytest
from fake_documents import FakeEmbedder, FakeStore
from mcp import Client, MCPError

from clm_decision import derived
from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server

HANDBOOK = Path(__file__).parent.parent / "examples" / "handbook"
TARIFF = Path(__file__).parent.parent / "examples" / "tariff" / "environment.toml"
RULES = ["carryover_limit", "match_eligible", "parental_eligible", "tuition_eligible", "vesting"]


def store():
    found = FakeStore()
    found.replace("handbook", [json.loads(line) for line in (HANDBOOK / "handbook.jsonl").read_text().splitlines()],
                  [], {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "embeddinggemma",
                       "dimensions": 26})
    return found


def load(name="environment.toml"):
    return load_environment(HANDBOOK / name, store=store(), embedder=FakeEmbedder())


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def read(client, uri: str):
    result = await client.read_resource(uri)
    assert result.contents[0].mime_type == "application/json"
    return json.loads(result.contents[0].text)


@pytest.mark.anyio
async def test_describe_is_a_resource_with_what_the_tool_returns():
    async with Client(build_environment_server(load())) as client:
        tool = await client.call_tool("describe", {})
        assert await read(client, "environment://describe") == tool.structured_content


@pytest.mark.anyio
async def test_every_rule_is_listed_and_reads_as_the_rule_tool_returns_it():
    async with Client(build_environment_server(load())) as client:
        listed = {str(resource.uri) for resource in (await client.list_resources()).resources}
        templates = {template.uri_template for template in (await client.list_resource_templates()).resource_templates}
        tool = await client.call_tool("rule", {"name": "match_eligible"})
        found = await read(client, "rule://match_eligible")
    assert {f"rule://{name}" for name in RULES} | {"environment://describe"} <= listed
    assert "rule://{name}" in templates
    assert found == tool.structured_content and found["origin"] == "supplied"


@pytest.mark.anyio
async def test_an_unknown_rule_says_which_rules_there_are():
    async with Client(build_environment_server(load())) as client:
        with pytest.raises(MCPError, match=r"no rule 'nope'; the rules are: carryover_limit, match_eligible"):
            await client.read_resource("rule://nope")


@pytest.mark.anyio
async def test_the_derived_layer_is_a_resource_only_when_the_environment_has_one():
    async with Client(build_environment_server(load("environment_derived.toml"))) as client:
        found = await read(client, "derived://layer")
    assert found == derived.read(HANDBOOK) and found["decisions"]
    async with Client(build_environment_server(load())) as client:
        listed = {str(resource.uri) for resource in (await client.list_resources()).resources}
    assert "derived://layer" not in listed


@pytest.mark.anyio
async def test_an_environment_without_rules_lists_no_rules():
    async with Client(build_environment_server(load_environment(TARIFF))) as client:
        listed = {str(resource.uri) for resource in (await client.list_resources()).resources}
        templates = {template.uri_template for template in (await client.list_resource_templates()).resource_templates}
    assert listed == {"environment://describe"} and templates == set()
