"""Tests for examples/noncompete: the three questions over its real statutes and made-up cases, with a stand-in
store, embedder and CLM, so no PostgreSQL, Ollama or model is needed."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument
import json
from pathlib import Path

import pytest
import judge_labels
import measure
import noncompete_functions
from fake_documents import FakeEmbedder, FakeStore
from mcp import Client

from clm_decision import decision_record
from clm_decision.engine import run_decision, run_query, run_scenario
from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server

EXAMPLE = Path(__file__).parent.parent / "examples" / "noncompete"
CASES = Path(__file__).parent / "data" / "noncompete" / "cases.jsonl"
BUILT = {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "embeddinggemma", "dimensions": 26}
TEXAS = {"state": "TX", "role": "sales manager", "served": "Texas, Oklahoma and Louisiana"}


def store():
    found = FakeStore()
    for name, path in (("statutes", EXAMPLE / "statutes.jsonl"), ("cases", CASES)):
        found.replace(name, [json.loads(line) for line in path.read_text().splitlines()], [], BUILT)
    return found


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def env(fake_clm):
    fake_clm.probabilities = {"decline to follow": 0.9, "anywhere in the United States": 0.85}
    return load_environment(EXAMPLE / "environment.toml", store=store(), embedder=FakeEmbedder())


def branch(result, **levers):
    return next(b for b in result["branches"] if b["levers"] == levers)


def test_california_is_void_with_its_citation(env):
    result = run_scenario(env, "agreement_by_state", {"agreement": "standard_v3"})
    california = branch(result, state="CA")
    assert california["status"] == "rejected" and california["figures"]["holds"] == "void"
    assert california["figures"]["citation"] == "ca-bpc-16600; ca-bpc-16600.1; ca-bpc-16600.5"
    [decision] = [s for s in california["steps"] if s["kind"] == "decision"]
    assert decision["citations"][0]["citation"] == "Cal. Bus. & Prof. Code § 16600"


def test_a_rules_citation_carries_its_collection_date_and_passages(env):
    result = run_scenario(env, "agreement_by_state", {"agreement": "standard_v3"})
    [decision] = [s for s in branch(result, state="CA")["steps"] if s["kind"] == "decision"]
    assert decision["citations"][0]["built_at"] == BUILT["built_at"]
    assert "(a)" in [passage["label"] for passage in decision["citations"][0]["passages"]]
    call = {"tool": "scenario_agreement_by_state", "arguments": {}, "result": result, "error": None}
    stored = decision_record.build("Does it hold?", [call])["documents"]
    assert {"collection": "statutes", "built_at": BUILT["built_at"], "document": "ca-bpc-16600",
            "citation": "Cal. Bus. & Prof. Code § 16600", "passage": "(a)"} in [
                {k: v for k, v in item.items() if k != "text"} for item in stored]


def test_colorado_holds_only_above_the_threshold(env):
    colorado = branch(run_scenario(env, "agreement_by_state", {"agreement": "standard_v3"}), state="CO")
    assert colorado["status"] == "rejected" and colorado["figures"]["holds"] == "only above threshold"
    assert colorado["rejected_by"][0]["reason"].startswith("Void for workers earning below Colorado's threshold")


def test_every_state_has_a_verdict(env):
    result = run_scenario(env, "agreement_by_state", {"agreement": "standard_v3"})
    holds = {b["levers"]["state"]: b["figures"]["holds"] for b in result["branches"]}
    assert holds == {"TX": "likely, if reasonable", "CA": "void", "NY": "case by case", "FL": "likely",
                     "IL": "only above threshold", "CO": "only above threshold", "WA": "only above threshold",
                     "MN": "void", "MA": "with changes", "OK": "void", "ND": "void"}


def test_a_colorado_worker_is_governed_by_colorado_law(env):
    result = run_decision(env, "choice_of_law", {"resides": "CO", "works": "CO"})
    assert result["outputs"]["governing"] == "CO" and result["outputs"]["citation"] == "co-crs-8-2-113"
    assert run_decision(env, "choice_of_law", {"resides": "TX", "works": "TX"})["outputs"]["governing"] == "contract"


def test_the_employee_query_takes_the_parameter_it_declares(env):
    [params] = [q.params for name, q in env.queries.items() if name == "employee"]
    result = run_query(env, "employee", {name: "e_1042" for name in params})
    assert result["rows"][0]["resides"] == "CO"


def test_narrowing_keeps_the_broadest_terms_that_pass(env):
    result = run_scenario(env, "narrow_to_enforce", TEXAS)
    assert branch(result, length_months=24, area="nationwide")["status"] == "rejected"
    assert result["best"] == {"length_months": 18, "area": "served"}


def test_the_bottom_line_is_calculated_from_the_cases(env):
    best = branch(run_scenario(env, "narrow_to_enforce", TEXAS), length_months=18, area="served")["figures"]
    assert best["similar"] == 6 and best["as_broad"] == 4
    assert best["enforced_as_written"] == 2 and best["share_enforced"] == 0.5
    assert {case["weight"] for case in best["cases"]} == {"binding", "strong"}


def test_a_similar_case_is_the_same_line_of_work_in_a_court_of_weight(env):
    best = branch(run_scenario(env, "narrow_to_enforce", TEXAS), length_months=18, area="served")["figures"]
    # made-3 is an account executive, made-8 a federal court's: neither counts
    assert {case["document"] for case in best["cases"]} == {"made-1", "made-2", "made-4", "made-5", "made-6",
                                                            "made-7"}
    assert best["broader_than_needed"] == 0.1


@pytest.mark.parametrize("area", [
    "within a radius of 250 miles of any office of the Company or Huntingdon International Holdings",
    "within fifty miles of the location where he was employed, or within ten miles of any other AutoNation dealer",
    "within a fifty-mile radius of the dealership", "Southeastern United States",
    "the entire western half of the country", "in Texas", None])
def test_a_regional_area_is_not_nationwide(area):
    assert not noncompete_functions._nationwide(area)  # pylint: disable=protected-access


@pytest.mark.parametrize("area", [
    "nationwide", "anywhere in the United States", "continental United States", "the entire United States",
    "anywhere in the country where IBS conducts business", "THE ENTIRE UNITED STATES OF AMERICA"])
def test_a_nationwide_area_is_nationwide(area):
    assert noncompete_functions._nationwide(area)  # pylint: disable=protected-access


@pytest.mark.parametrize("asked, found, same", [
    ("sales manager", "regional sales manager", True), ("sales manager", "salesman", True),
    ("senior sales engineer", "regional sales manager", True), ("senior sales engineer", "senior accountant", False),
    ("general manager", "general partner", False), ("vice president", "customer service manager", False),
    ("", "sales manager", False), ("senior", "senior accountant", False), ("sales manager", None, False)])
def test_the_line_of_work_skips_generic_first_words(asked, found, same):
    assert noncompete_functions._same_line_of_work(asked, found) is same  # pylint: disable=protected-access


def test_still_good_law_reads_the_citing_passages(env):
    best = branch(run_scenario(env, "narrow_to_enforce", TEXAS), length_months=18, area="served")["figures"]
    binding = next(case for case in best["cases"] if case["weight"] == "binding")
    assert [(c["document"], c["departs"]) for c in binding["citing"]] == [("made-9", 0.9)]
    assert "most_critical_citing" not in binding


def test_no_rule_reads_a_search_score(env):
    result = run_scenario(env, "narrow_to_enforce", TEXAS)
    for item in result["branches"]:
        for step in item["steps"]:
            if step["kind"] in ("decision", "calculation"):
                assert "score" not in json.dumps(step["inputs"])


@pytest.mark.anyio
async def test_the_environment_without_documents_offers_no_document_tools(fake_clm):
    env = load_environment(EXAMPLE / "environment_without_documents.toml", store=store(), embedder=FakeEmbedder())
    async with Client(build_environment_server(env)) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
    assert "scenario_narrow_to_enforce" in names and not [n for n in names if n.startswith(("search_", "document_"))]


def test_every_result_says_not_legal_advice_and_what_it_cannot_tell(env):
    assert env.description.startswith("NOT LEGAL ADVICE")
    by_state = run_scenario(env, "agreement_by_state", {"agreement": "standard_v3"})
    narrowed = run_scenario(env, "narrow_to_enforce", TEXAS)
    for item in by_state["branches"] + narrowed["branches"]:
        if item["figures"]:
            assert item["figures"]["notice"].startswith("Not legal advice")
    assert all("cannot tell" in item["figures"]["cannot_tell"] for item in narrowed["branches"] if item["figures"])
    search = next(s for s in narrowed["branches"][0]["steps"] if s["kind"] == "search")
    assert search["built_at"] == "2026-11-03T12:00:00Z"


def test_judge_labels_measures_each_judge_and_checks_the_courts_words(env):
    labels = [{"text": "We decline to follow Made-up Co. v. Seller 6", "yes": True, "document": "made-9",
               "passage": "1"},
              {"text": "The covenant was reasonable.", "yes": False, "document": "made-6", "passage": "1"}]
    judged = judge_labels.measure(env, "departs", labels)
    assert [p for p, _ in judged] == [0.9, 0.1]
    assert judge_labels.separation(judged) == (0.9, 0.1, 0.5)
    assert judge_labels.separation([(0.4, labels[0]), (0.6, labels[1])])[2] is None
    assert judge_labels.unquoted(env, labels) == [labels[1]]


def test_measure_counts_the_keys_citations_by_section_and_reporter():
    cited = ["Tex. Bus. & Com. Code § 15.51(c)", "793 S.W.2d 302", "Tex. Bus. & Com. Code § 15.50(a)"]
    answer = "Under § 15.51 a court reforms it (Webb v. Hartman, 793 S.W. 2d 302)."
    assert measure.key_cited(answer, cited) == 2
    key = measure._key()  # pylint: disable=protected-access
    assert measure.expected(key, "q3") == {"governing": "CO", "holds": "only above threshold"}
    assert measure.right({"TX": "likely, if reasonable", "CA": "likely"}, measure.expected(key, "q1")) == 1
