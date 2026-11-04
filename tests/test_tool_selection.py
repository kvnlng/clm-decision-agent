"""Tests for choosing an environment server's tools, and a read-only one: what is not offered is gone from the tool
list, describe, the resources and the prompt, and the scenarios still run. The handbook, with a stand-in store."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import dataclasses
import shutil

import pytest
from mcp import Client, MCPError
from test_environment_resources import HANDBOOK, copied, load, read

from clm_decision import instructions, server
from clm_decision.environment_server import build_environment_server
from clm_decision.scenario import ScenarioError
from clm_decision.serving import Serve

HANDBOOK_TOOLS = ("search_handbook", "document_handbook", "cites_handbook", "cited_by_handbook")
FEWER = Serve(tools=("describe", "scenario_*", "search_*", "document_*"))


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def listing(server):
    async with Client(server) as client:
        tools = sorted(tool.name for tool in (await client.list_tools()).tools)
        resources = {str(resource.uri) for resource in (await client.list_resources()).resources}
        templates = {template.uri_template for template in (await client.list_resource_templates()).resource_templates}
        described = (await client.call_tool("describe", {})).structured_content
        resource = await read(client, "environment://describe")
    return tools, resources | templates, described, resource


@pytest.mark.anyio
async def test_only_the_chosen_tools_are_listed():
    tools, _, _, _ = await listing(build_environment_server(load(), serve=FEWER))
    assert tools == ["describe", "document_handbook", "scenario_carryover", "scenario_match",
                     "scenario_parental_leave", "scenario_tuition", "search_handbook"]


@pytest.mark.anyio
async def test_describe_holds_only_what_is_offered_and_names_what_was_left_out():
    _, _, described, resource = await listing(build_environment_server(load(), serve=FEWER))
    assert described == resource
    assert (described["queries"], described["decisions"], described["calculations"]) == ({}, {}, {})
    assert set(described["scenarios"]) == {"match", "carryover", "parental_leave", "tuition"}
    assert "topics" not in described and "handbook" in described["collections"]
    assert "query_employee" in described["left_out"] and "rules" in described["left_out"]
    assert "cites_handbook" in described["left_out"] and "read_only" not in described


@pytest.mark.anyio
async def test_by_default_describe_is_as_before():
    _, _, described, _ = await listing(build_environment_server(load()))
    assert "left_out" not in described and "read_only" not in described
    assert described["topics"] == ["benefits"] and "employee" in described["queries"]


@pytest.mark.anyio
async def test_a_collection_with_none_of_its_tools_is_not_described():
    _, uris, described, _ = await listing(build_environment_server(load(), serve=Serve(exclude_tools=HANDBOOK_TOOLS)))
    assert "collections" not in described
    assert not any(uri.startswith("document://") for uri in uris)


@pytest.mark.anyio
async def test_rule_resources_go_with_the_rule_tool():
    _, uris, described, _ = await listing(build_environment_server(load(), serve=Serve(exclude_tools=("rule",))))
    assert not any(uri.startswith("rule://") for uri in uris)
    assert described["topics"] == ["benefits"]


@pytest.mark.anyio
async def test_document_resources_go_with_the_document_tool():
    _, uris, _, _ = await listing(build_environment_server(load(), serve=Serve(exclude_tools=("document_handbook",))))
    assert not any(uri.startswith("document://") for uri in uris)


@pytest.mark.anyio
async def test_without_a_document_tool_the_derived_resource_holds_no_quote(tmp_path):
    env = copied(tmp_path, {})
    async with Client(build_environment_server(env, serve=Serve(exclude_tools=("document_*",)))) as client:
        found = await read(client, "derived://in-force")
    assert found["entries"] and all("quote" not in entry["source"] for entry in found["entries"].values())


@pytest.mark.anyio
async def test_a_scenario_runs_when_its_queries_and_decisions_are_not_offered():
    env = load()
    arguments = {"employee": "E-1002", "as_of": "2026-12-01"}
    async with Client(build_environment_server(env)) as client:
        whole = (await client.call_tool("scenario_match", arguments)).structured_content
    async with Client(build_environment_server(env, serve=FEWER)) as client:
        fewer = await client.call_tool("scenario_match", arguments)
    assert not fewer.is_error and fewer.structured_content == whole


@pytest.mark.anyio
async def test_the_prompt_names_only_tools_that_are_offered():
    env = load()
    question = "Am I eligible for the match?"
    async with Client(build_environment_server(env, serve=Serve(exclude_tools=("rules", "scenario_*")))) as client:
        text = (await client.get_prompt("answer_question", {"question": question})).messages[0].content.text
    assert "call rules" not in text and "run the scenario tool" not in text
    assert text.startswith("You answer") and "Start with describe" in text and text.endswith(question)
    assert instructions.answer_question(env.name, env.description, question, None) == \
        instructions.answer_question(env.name, env.description, question)


@pytest.mark.anyio
async def test_the_servers_instructions_name_only_tools_that_are_offered():
    async with Client(build_environment_server(load(), serve=Serve(exclude_tools=("scenario_*",)))) as client:
        said = client.instructions
    assert "Call describe first" in said and "run a scenario" not in said


@pytest.mark.anyio
async def test_without_a_selection_the_instructions_are_as_before_even_with_no_scenarios():
    async with Client(build_environment_server(load("environment_documents_only.toml"))) as client:
        said = client.instructions
    assert said == ("Decisions for 'Employee handbook (documents only)'. Call describe first. Every figure and verdict "
                    "comes from these tools: run a scenario to compare alternatives rather than combining queries and "
                    "decisions yourself.")


@pytest.mark.anyio
async def test_read_only_changes_no_tool_and_describe_says_so():
    tools, _, described, _ = await listing(build_environment_server(load(), serve=Serve(read_only=True)))
    plain, _, _, _ = await listing(build_environment_server(load()))
    assert tools == plain and described["read_only"] is True and "left_out" not in described


def test_an_unknown_tool_is_one_line_naming_the_tools():
    with pytest.raises(ScenarioError, match=r"^tools: 'query_nothing' matches no tool; the tools are: query_employee,"):
        build_environment_server(load(), serve=Serve(tools=("query_nothing",)))


def test_clm_server_reads_an_environments_serve_table(tmp_path, capsys):
    folder = tmp_path / "tariff"
    shutil.copytree(HANDBOOK.parent / "tariff", folder, ignore=shutil.ignore_patterns("runs"))
    toml = folder / "environment.toml"
    toml.write_text(toml.read_text() + '\n[serve]\ntools = ["search_*"]\n')
    assert server.main([str(toml)]) == 2
    assert capsys.readouterr().err.strip().startswith("error: tools: 'search_*' matches no tool; the tools are: ")


def with_secret(env):
    """The environment with a second collection, `secret`, holding one document."""
    env.store.replace("secret", [{"id": "secret-1", "title": "Secret", "citation": "Secret § 1", "metadata": {},
                                  "passages": [{"label": "(1)", "text": "Hidden."}]}], [],
                      env.store.built("handbook"))
    return dataclasses.replace(env, collections={**env.collections, "secret": env.collections["handbook"]})


@pytest.mark.anyio
async def test_a_left_out_collection_cannot_be_read_through_the_templates():
    env = with_secret(load())
    async with Client(build_environment_server(env)) as client:
        assert (await read(client, "document://secret/secret-1"))["document"]["id"] == "secret-1"
    async with Client(build_environment_server(env, serve=Serve(tools=("describe", "document_handbook")))) as client:
        for uri in ("document://secret/secret-1", "document://secret/secret-1/(1)"):
            with pytest.raises(MCPError, match=r"no collection 'secret' \(the environment has: handbook\)"):
                await client.read_resource(uri)
        assert (await read(client, "document://handbook/hb-welcome"))["document"]["id"] == "hb-welcome"


@pytest.mark.anyio
async def test_derived_quotes_come_only_from_collections_whose_document_tool_is_offered(tmp_path):
    env = with_secret(copied(tmp_path, {}))
    async with Client(build_environment_server(env, serve=Serve(exclude_tools=("document_handbook",)))) as client:
        found = await read(client, "derived://in-force")
    assert found["entries"] and all("quote" not in entry["source"] for entry in found["entries"].values())
    async with Client(build_environment_server(env, serve=Serve(exclude_tools=("document_secret",)))) as client:
        found = await read(client, "derived://in-force")
    assert all(entry["source"]["quote"] for entry in found["entries"].values())
