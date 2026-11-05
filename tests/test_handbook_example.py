"""Tests for examples/handbook: Meridian's made-up handbook, employees and rules, checked against the answer key
written by hand, with a stand-in store and embedder, so no PostgreSQL or Ollama is needed."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import importlib.util
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fake_documents import FakeEmbedder, FakeStore
from mcp import Client

from clm_decision import serving
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


def _measure():
    spec = importlib.util.spec_from_file_location("handbook_measure", EXAMPLE / "measure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KEYED = {"figures": {"match": {"value": 3400, "about": "a"}, "cap": {"value": 6.0, "about": "b"},
                     "limit": {"value": 24500, "about": "c"}}, "superseded": [1840]}


def item(value, hedged=False):
    return {"value": value, "hedged": hedged}


def test_a_right_figure_a_wrong_one_and_a_missing_one():
    read = {"items": {"match": item(3400), "cap": item(5.0), "limit": item(None)}, "other": []}
    assert _measure().score_figures(read, KEYED) == {"right": 1, "wrong": 1, "superseded": 0, "not_given": 1,
                                                     "hedged": 0, "other_from_tools": 0, "other_unsourced": 0,
                                                     "wrong_as_fact": 1}


def test_the_old_match_is_wrong_and_counted_as_superseded():
    read = {"items": {"match": item(1840), "cap": item(6), "limit": item(24500)}, "other": []}
    scored = _measure().score_figures(read, KEYED)
    assert scored["wrong"] == 1 and scored["superseded"] == 1 and scored["right"] == 2


def test_a_hedged_guess_is_not_wrong():
    read = {"items": {"match": item(2000, hedged=True), "cap": item(None), "limit": item(None)},
            "other": [{"text": "typically 50% of 6%", "value": 50, "hedged": True}]}
    scored = _measure().score_figures(read, KEYED)
    assert scored["hedged"] == 2 and scored["wrong_as_fact"] == 0 and scored["not_given"] == 2


def test_a_further_figure_is_counted_by_whether_a_tool_returned_it():
    read = {"items": {"match": item(3400), "cap": item(6), "limit": item(24500)},
            "other": [{"text": "the match at 3% is $2,760", "value": 2760, "hedged": False},
                      {"text": "a combined limit of $72,000", "value": 72000, "hedged": False}]}
    scored = _measure().score_figures(read, KEYED, [{"match": 2760}])
    assert scored["other_from_tools"] == 1 and scored["other_unsourced"] == 1
    assert scored["wrong_as_fact"] == 0 and scored["right"] == 3


def test_a_further_figure_no_tool_returned_is_unsourced():
    # 1250000 was once written as 1.25e+06
    read = {"items": {"match": item(3400), "cap": item(6), "limit": item(24500)},
            "other": [{"text": f"figure {value}", "value": value, "hedged": False}
                      for value in (5, 4.0, 2030, 1250000, 777)]}
    scored = _measure().score_figures(read, KEYED, ["nothing here"])
    assert scored["other_from_tools"] == 0 and scored["other_unsourced"] == 5


def test_a_large_further_figure_a_tool_returned_is_from_tools():
    read = {"items": {"match": item(3400), "cap": item(6), "limit": item(24500)},
            "other": [{"text": "$1,250,000", "value": 1250000, "hedged": False}]}
    assert _measure().score_figures(read, KEYED, [{"total": 1250000}])["other_from_tools"] == 1


def test_a_small_further_figure_or_year_a_tool_returned_is_from_tools():
    # number_check skips small whole numbers and years, but the tool's value is still compared
    read = {"items": {"match": item(3400), "cap": item(6), "limit": item(24500)},
            "other": [{"text": "6 weeks", "value": 6, "hedged": False},
                      {"text": "in 2030", "value": 2030, "hedged": False},
                      {"text": "$5", "value": 5, "hedged": False}]}
    scored = _measure().score_figures(read, KEYED, [{"weeks": 6}, "the plan ends in 2030"])
    assert scored["other_from_tools"] == 2 and scored["other_unsourced"] == 1


class Recorder:
    """A stand-in Anthropic client that keeps the request and answers nothing given."""

    def __init__(self):
        self.request, self.messages = None, self

    def create(self, **request):
        self.request = request
        reply = {"items": {name: item(None) for name in KEY["q1"]["figures"]}, "other": [], "verdict": "not given"}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(reply))])


def test_the_reader_is_never_shown_the_keys_values():
    client = Recorder()
    read = _measure().read_figures(client, "I don't know Meridian's plan.", KEY["q1"])
    sent = json.dumps(client.request)
    assert read["verdict"] == "not given" and "the company match in dollars" in sent
    assert not [f for f in KEY["q1"]["figures"].values() if f"{f['value']:,}" in sent or f"{f['value']}" in sent]


def test_policy_cited_counts_the_keys_sections_by_section_number():
    measure = _measure()
    assert measure.policy_cited("See Handbook § 4(3).", KEY["q1"]) == 1
    assert measure.policy_cited("See the Tuition assistance policy.", KEY["q4"]) == 0
    assert measure.policy_cited("Handbook §7(2)", KEY["q4"]) == 1
    assert measure.policy_cited("See § 7.", KEY["q1"]) == 0


@pytest.mark.anyio
async def test_the_fewer_tools_setup_is_the_handbook_with_only_describe_scenarios_and_documents(tmp_path):
    path = _measure().fewer_tools(tmp_path)
    assert path.parent.parent == tmp_path and path.name == "environment.toml"
    env = load_environment(path, store=store(), embedder=FakeEmbedder())
    assert env.description == load().description
    async with Client(build_environment_server(env, serve=serving.read(path))) as client:
        names = sorted(tool.name for tool in (await client.list_tools()).tools)
    assert names == ["describe", "document_handbook", "scenario_carryover", "scenario_match",
                     "scenario_parental_leave", "scenario_tuition", "search_handbook"]
