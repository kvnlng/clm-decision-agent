"""Tests for examples/handbook: Meridian's made-up handbook, employees and rules, checked against the answer key
written by hand, with a stand-in store and embedder, so no PostgreSQL or Ollama is needed."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json
import tomllib
from pathlib import Path

import pytest
from fake_documents import FakeEmbedder, FakeStore
from mcp import Client

from clm_decision.engine import Steps, run_scenario
from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server

EXAMPLE = Path(__file__).parent.parent / "examples" / "handbook"
KEY = tomllib.loads((EXAMPLE / "answer_key.toml").read_text(encoding="utf-8"))
TODAY = "2026-12-01"
# Each question's scenario, its parameters beyond the employee and date, and the branch the key describes
ASKED = {"q1": ("match", {}, {"contribution_percent": 3}),
         "q2": ("carryover", {}, {"days_taken": 0}),
         "q3": ("parental_leave", {}, {"weeks_taken": 15}),
         "q4": ("tuition", {"tuition": 3900, "job_related": True}, {"calendar_year": 2027})}
YES = {"q1": "eligible for the match", "q2": "yes, part of it", "q3": "eligible", "q4": "reimbursed"}


def store():
    found = FakeStore()
    found.replace("handbook", [json.loads(line) for line in (EXAMPLE / "handbook.jsonl").read_text().splitlines()],
                  [], {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "embeddinggemma",
                       "dimensions": 26})
    return found


def load(name="environment.toml"):
    return load_environment(EXAMPLE / name, store=store(), embedder=FakeEmbedder())


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def env():
    return load()


def ask(env, question, employee=None, as_of=TODAY):
    scenario, params, _ = ASKED[question]
    return run_scenario(env, scenario, {"employee": employee or KEY[question]["employee"], "as_of": as_of, **params})


def branch(result, levers):
    return next(b for b in result["branches"] if b["levers"] == levers)


@pytest.mark.parametrize("question", ["q1", "q2", "q3", "q4"])
def test_each_question_gets_the_keys_verdict_and_figures(env, question):
    found = branch(ask(env, question), ASKED[question][2])
    assert found["status"] == "kept" and not found["rejected_by"]
    assert KEY[question]["verdict"] == YES[question]
    for item, figure in KEY[question]["figures"].items():
        assert found["figures"][item] == pytest.approx(figure["value"]), item


def test_the_match_comes_from_the_amendment(env):
    result = ask(env, "q1")
    assert branch(result, {"contribution_percent": 4})["figures"]["company_match"] == 3220
    assert result["best"]["contribution_percent"] in (6, 8)
    text = json.dumps(result)
    assert not [value for value in KEY["q1"]["superseded"] if value and str(value) in text]


def test_eligibility_begins_on_the_day_the_handbook_says(env):
    def eligible(as_of):
        return not ask(env, "q1", "e_2007", as_of)["branches"][0]["rejected_by"]
    assert eligible("2026-12-01") and not eligible("2026-11-30")
    assert all(b["rejected_by"] for b in ask(env, "q1", "e_2006")["branches"])


def test_an_unknown_employee_is_an_error_naming_it(env):
    result = ask(env, "q1", "e_9999")
    assert all(b["status"] == "failed" and "e_9999" in b["error"] for b in result["branches"])


def test_the_superseded_summary_says_when_it_ended(env):
    rows = Steps(env).search("handbook", "company match on the 401(k)", None, 13)
    old = next(row for row in rows if row["document"] == "hb-401k-2024")
    assert old["metadata"]["effective_until"] == "2025-12-31"


def test_the_rules_only_results_hold_no_passage_text():
    bare = load("environment_without_documents.toml")
    for question in ASKED:
        result = ask(bare, question)
        assert all(b["status"] != "failed" for b in result["branches"])
        assert '"passages"' not in json.dumps(result)
    assert '"passages"' in json.dumps(ask(load(), "q1"))


@pytest.mark.anyio
async def test_the_environment_without_documents_offers_no_document_tools():
    async with Client(build_environment_server(load("environment_without_documents.toml"))) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
    assert "scenario_match" in names and not [n for n in names if n.startswith(("search_", "document_"))]


def test_every_result_says_not_advice(env):
    assert env.description.startswith("NOT HR, TAX OR LEGAL ADVICE")
    for question in ASKED:
        for found in ask(env, question)["branches"]:
            assert found["figures"]["notice"].startswith("Not HR, tax or legal advice")
