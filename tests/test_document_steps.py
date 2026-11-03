"""Tests for the document steps in engine.py and their tools in environment_server.py: search, read a document,
follow citations; each traced like a query."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
from conftest import ENV_TOML
from fake_documents import FakeEmbedder, FakeStore
from mcp import Client

from clm_decision import engine
from clm_decision.engine import StepError, Steps
from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server, describe

DOCUMENTS = '''
[documents]
password_env = "TOY_DOCUMENTS_PASSWORD"

[collections.cases]
description = "Court opinions on non-competes"
embedding_model = "toy-embed"
filters = { state = "string", year = "integer" }
'''
CASES = [
    {"id": "case-a", "citation": "100 S.W.3d 1", "title": "Case A", "metadata": {"state": "TX", "year": 2019},
     "passages": [{"label": "1", "text": "We reform the covenant to the territory he served."}],
     "cites": [{"id": "tx-15.50", "passage": "1"}]},
    {"id": "tx-15.50", "citation": "Tex. Bus. & Com. Code § 15.50", "title": "Statute",
     "metadata": {"state": "TX", "year": 1989},
     "passages": [{"label": "(a)", "text": "A covenant is enforceable with reasonable limits."}], "cites": []},
]
BUILT = {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "toy-embed", "dimensions": 26}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def env(write_environment):
    store = FakeStore()
    store.replace("cases", CASES, [], BUILT)
    return load_environment(write_environment(ENV_TOML + DOCUMENTS), store=store, embedder=FakeEmbedder())


def test_a_search_is_a_traced_step(env):
    steps = Steps(env)
    rows = steps.search("cases", "reform the covenant", {"state": "TX"}, 5)
    assert rows[0]["document"] == "case-a" and rows[0]["citation"] == "100 S.W.3d 1"
    assert steps.steps == [{"kind": "search", "name": "cases",
                            "inputs": {"text": "reform the covenant", "filters": {"state": "TX"}, "limit": 5},
                            "result": rows, "built_at": "2026-11-03T12:00:00Z"}]
    assert env.embedder.asked == [("toy-embed", ["reform the covenant"])]


def test_an_empty_collection_names_the_loader(env):
    env.store.replace("cases", [], [], BUILT)
    with pytest.raises(StepError, match=r"^collection cases holds nothing: load it with clm-load-documents "
                                        r"environment\.toml cases FILE\.jsonl$"):
        Steps(env).search("cases", "covenant")


def test_a_different_embedding_model_is_one_line(env):
    env.store.info["cases"] = BUILT | {"model": "other-embed"}
    with pytest.raises(StepError, match=r"^collection cases was loaded with other-embed, but the environment names "
                                        r"toy-embed: load it again, or name other-embed$"):
        Steps(env).search("cases", "covenant")


def test_a_different_vector_size_is_one_line(env):
    env.store.info["cases"] = BUILT | {"dimensions": 768}
    with pytest.raises(StepError, match=r"^collection cases holds vectors of 768 numbers, but toy-embed gave 26"):
        Steps(env).search("cases", "covenant")


@pytest.mark.parametrize("filters, message", [
    ({"judge": "Smith"}, r"^search cases: judge is not a filter \(state, year\)$"),
    ({"year": "2019"}, r"^search cases: year must be a whole number, a list of them, or \{\"from\", \"to\"\}"),
    ({"state": {"from": "A"}}, r"^search cases: state is text: give a value or a list, not a range$"),
])
def test_an_undeclared_or_mistyped_filter_is_one_line(env, filters, message):
    with pytest.raises(StepError, match=message):
        Steps(env).search("cases", "covenant", filters)


def test_empty_text_and_a_bad_limit_are_refused(env):
    with pytest.raises(StepError, match="^search cases: there is no text to search for$"):
        Steps(env).search("cases", " ")
    with pytest.raises(StepError, match="^search cases: limit must be from 1 to 50"):
        Steps(env).search("cases", "covenant", limit=0)


def test_document_and_citations_are_steps(env):
    steps = Steps(env)
    assert steps.document("cases", "tx-15.50", "(a)")["passages"][0]["label"] == "(a)"
    assert steps.cited_by("cases", "tx-15.50") == [
        {"document": "case-a", "citation": "100 S.W.3d 1", "depth": 1, "passage": "1"}]
    assert [step["kind"] for step in steps.steps] == ["document", "citations"]
    assert steps.steps[1]["inputs"] == {"id": "tx-15.50", "depth": 1, "direction": "cited_by"}
    with pytest.raises(StepError, match="^collection cases: no document nothing$"):
        steps.document("cases", "nothing")
    with pytest.raises(StepError, match=r"^collection cases: tx-15\.50 has no passage \(z\)"):
        steps.document("cases", "tx-15.50", "(z)")


def test_describe_lists_collections_with_their_date(env):
    assert describe(env)["collections"] == {"cases": {
        "description": "Court opinions on non-competes", "filters": {"state": "string", "year": "integer"},
        "built": BUILT}}


@pytest.mark.anyio
async def test_each_collection_gets_four_tools(env):
    async with Client(build_environment_server(env)) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        assert {"search_cases", "document_cases", "cites_cases", "cited_by_cases"} <= names
        result = await client.call_tool("search_cases", {"text": "reform the covenant"})
    assert result.structured_content["rows"][0]["document"] == "case-a"
    assert result.structured_content["built_at"] == "2026-11-03T12:00:00Z"


def test_run_search_gives_rows_and_steps(env):
    result = engine.run_search(env, "cases", {"text": "covenant", "filters": None, "limit": 10})
    assert set(result) == {"rows", "built_at", "steps"}
