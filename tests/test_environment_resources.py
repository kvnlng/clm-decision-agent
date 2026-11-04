"""Tests for the environment server's resources and prompt: the same data as the read tools, through a second door,
with a stand-in store so no PostgreSQL or Ollama is needed."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import dataclasses
import json
import shutil
from pathlib import Path

import pytest
from fake_documents import FakeEmbedder, FakeStore
from mcp import Client, MCPError

from clm_decision import derived, environment_server, instructions, tracing
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


def copied(tmp_path, status_of: dict):
    """The handbook in tmp_path, with the named derived decisions given another status."""
    folder = tmp_path / "handbook"
    shutil.copytree(HANDBOOK, folder, ignore=shutil.ignore_patterns("runs"))
    layer = derived.read(folder)
    for name, status in status_of.items():
        layer["decisions"][name]["status"] = status
    derived.write(folder, layer)
    return load_environment(folder / "environment_derived.toml", store=store(), embedder=FakeEmbedder())


@pytest.mark.anyio
async def test_the_derived_resource_holds_only_the_entries_in_force(tmp_path):
    env = copied(tmp_path, {"vesting": "rejected", "tuition_eligible": "proposed"})
    async with Client(build_environment_server(env)) as client:
        found = await read(client, "derived://in-force")
        rule = await read(client, "rule://match_eligible")
    assert set(found["entries"]) == set(env.admitted)
    assert "vesting" not in found["entries"] and "tuition_eligible" not in found["entries"]
    match = found["entries"]["match_eligible"]
    assert match["origin"] == "derived" and match["status"] == "approved" and match["kind"] == "decisions"
    assert match["source"]["document"] and match["source"]["passages"] and match["source"]["quote"]
    assert found["admit"] == ["approved"]
    assert rule["origin"] == "derived" and rule["status"] == "approved" and rule["source"]["document"]
    async with Client(build_environment_server(load())) as client:
        listed = {str(resource.uri) for resource in (await client.list_resources()).resources}
    assert not any(uri.startswith("derived://") for uri in listed)


@pytest.mark.anyio
async def test_without_document_tools_the_derived_resource_holds_no_quote(tmp_path):
    env = dataclasses.replace(copied(tmp_path, {}), document_tools=False)
    async with Client(build_environment_server(env)) as client:
        found = await read(client, "derived://in-force")
    assert found["entries"] and all("quote" not in entry["source"] for entry in found["entries"].values())
    assert all(entry["source"]["document"] for entry in found["entries"].values())


@pytest.mark.anyio
async def test_a_resource_read_is_traced_like_a_tool_call_and_so_is_a_failed_one(tmp_path):
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    async with Client(build_environment_server(load(), trace=writer)) as client:
        await client.read_resource("rule://match_eligible")
        await client.read_resource("document://handbook/hb-welcome/(1)")
        with pytest.raises(MCPError):
            await client.read_resource("rule://nope")
    writer.close()
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [(record["tool"], record["outcome"]) for record in records] == [
        ("rule://match_eligible", {"ok": True}), ("document://handbook/hb-welcome/(1)", {"ok": True}),
        ("rule://nope", {"error": "ResourceError"})]
    assert all(record["ms"] >= 0 for record in records)


@pytest.mark.anyio
async def test_rule_resources_are_offered_only_with_the_rule_tool():
    env = load()
    decisions = {name: dataclasses.replace(decision, rule=dataclasses.replace(decision.rule, governs=()))
                 if decision.rule else decision for name, decision in env.decisions.items()}
    env = dataclasses.replace(env, decisions=decisions)
    assert not env.topics()
    async with Client(build_environment_server(env)) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        listed = {str(resource.uri) for resource in (await client.list_resources()).resources}
        templates = {template.uri_template for template in (await client.list_resource_templates()).resource_templates}
    assert "rule" not in tools
    assert not any(uri.startswith("rule://") for uri in listed | templates)


@pytest.mark.anyio
async def test_an_environment_without_rules_lists_no_rules():
    async with Client(build_environment_server(load_environment(TARIFF))) as client:
        listed = {str(resource.uri) for resource in (await client.list_resources()).resources}
        templates = {template.uri_template for template in (await client.list_resource_templates()).resource_templates}
    assert listed == {"environment://describe"} and templates == set()


@pytest.mark.anyio
async def test_a_document_and_a_passage_read_as_the_document_tool_returns_them():
    async with Client(build_environment_server(load())) as client:
        whole = await client.call_tool("document_handbook", {"id": "hb-401k-2026"})
        one = await client.call_tool("document_handbook", {"id": "hb-401k-2026", "passage": "(2)"})
        assert await read(client, "document://handbook/hb-401k-2026") == whole.structured_content
        assert await read(client, "document://handbook/hb-401k-2026/(2)") == one.structured_content
        assert await read(client, "document://handbook/hb-401k-2026/%282%29") == one.structured_content
        with pytest.raises(MCPError, match=r"collection handbook: no document hb-nothing"):
            await client.read_resource("document://handbook/hb-nothing")


@pytest.mark.anyio
async def test_a_small_collection_lists_each_document_and_the_template_says_the_limit():
    async with Client(build_environment_server(load())) as client:
        resources = {str(resource.uri): resource for resource in (await client.list_resources()).resources}
        templates = {template.uri_template: template
                     for template in (await client.list_resource_templates()).resource_templates}
        described = await client.call_tool("describe", {})
    documents = sorted(uri for uri in resources if uri.startswith("document://"))
    assert len(documents) == 13 and "document://handbook/hb-welcome" in documents
    assert resources["document://handbook/hb-401k-2026"].description.startswith("Meridian")
    assert {"document://{collection}/{id}", "document://{collection}/{id}/{passage}"} <= set(templates)
    note = templates["document://{collection}/{id}"].description
    assert f"at most {environment_server.LISTED_DOCUMENTS} documents" in note
    assert set(described.structured_content["collections"]["handbook"]) == {"description", "filters", "built"}


@pytest.mark.anyio
async def test_a_large_collection_is_not_listed_but_can_be_read(monkeypatch):
    monkeypatch.setattr(environment_server, "LISTED_DOCUMENTS", 12)
    async with Client(build_environment_server(load())) as client:
        listed = {str(resource.uri) for resource in (await client.list_resources()).resources}
        found = await read(client, "document://handbook/hb-welcome")
    assert not any(uri.startswith("document://") for uri in listed) and found["document"]["id"] == "hb-welcome"


@pytest.mark.anyio
async def test_a_collection_loaded_with_another_model_is_not_listed_and_reading_says_why():
    other = store()
    other.replace("handbook", [json.loads(line) for line in (HANDBOOK / "handbook.jsonl").read_text().splitlines()],
                  [], {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "other-model",
                       "dimensions": 26})
    env = load_environment(HANDBOOK / "environment.toml", store=other, embedder=FakeEmbedder())
    async with Client(build_environment_server(env)) as client:
        listed = {str(resource.uri) for resource in (await client.list_resources()).resources}
        with pytest.raises(MCPError, match="was loaded with other-model"):
            await client.read_resource("document://handbook/hb-welcome")
    assert not any(uri.startswith("document://") for uri in listed)


@pytest.mark.anyio
async def test_no_document_tools_means_no_document_resources():
    env = dataclasses.replace(load(), document_tools=False)
    async with Client(build_environment_server(env)) as client:
        listed = {str(resource.uri) for resource in (await client.list_resources()).resources}
        templates = {template.uri_template for template in (await client.list_resource_templates()).resource_templates}
        described = await read(client, "environment://describe")
    assert not any(uri.startswith("document://") for uri in listed | templates)
    assert "resources" not in described["collections"]["handbook"]


@pytest.mark.anyio
@pytest.mark.parametrize("arguments", [None, {}, {"question": ""}, {"question": "  \n"}])
async def test_the_prompt_refuses_a_missing_or_blank_question_in_one_line_without_a_traceback(arguments, caplog):
    async with Client(build_environment_server(load())) as client:
        with pytest.raises(MCPError) as refused:
            await client.get_prompt("answer_question", arguments)
    assert str(refused.value.message) == "answer_question needs a question: give it the question to ask"
    assert not [record for record in caplog.records if record.exc_info or record.levelname == "ERROR"]


def test_the_fake_store_lists_ids_in_order_up_to_the_limit():
    assert store().ids("handbook", 3) == ["hb-401k-2024", "hb-401k-2026", "hb-dress"]
    assert store().ids("nothing", 3) == []


@pytest.mark.anyio
async def test_the_prompt_is_the_agents_instructions_with_the_environment_and_the_question():
    env = load()
    async with Client(build_environment_server(env)) as client:
        prompts = (await client.list_prompts()).prompts
        got = await client.get_prompt("answer_question", {"question": "Am I eligible for the match?"})
    assert [prompt.name for prompt in prompts] == ["answer_question"]
    assert [(argument.name, argument.required) for argument in prompts[0].arguments] == [("question", True)]
    assert len(got.messages) == 1 and got.messages[0].role == "user"
    text = got.messages[0].content.text
    assert text == instructions.answer_question(env.name, env.description, "Am I eligible for the match?")
    assert text.startswith(instructions.agent_system() + "\n\n")
    assert f"'{env.name}'" in text and env.description in text and text.endswith("Am I eligible for the match?")
