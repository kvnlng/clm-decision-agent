"""Tests for citation_check.py, for a decision table's citations resolved in its step, and for citations in the
decision record."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
from conftest import ENV_TOML
from fake_documents import FakeEmbedder, FakeStore
from test_rule_citations import DOCUMENTS, cited_dmn, with_collection

from clm_decision import citation_check, decision_record
from clm_decision.engine import Steps
from clm_decision.environment import load_environment

SEARCHED = [{"rows": [
    {"document": "tx-15.51", "citation": "Tex. Bus. & Com. Code § 15.51", "title": "Procedures", "metadata": {},
     "passage": "(c)", "text": "The court shall reform the covenant.", "score": 1.0},
    {"document": "cl-1", "citation": "100 S.W.3d 1", "title": "Case A", "metadata": {}, "passage": "2",
     "text": "We reform it.", "score": 0.5}]},
    {"outputs": {"allowed": False, "citation": "tx-15.50"},
     "citations": [{"id": "tx-15.50", "citation": "Tex. Bus. & Com. Code § 15.50", "title": "Criteria"}]}]


@pytest.mark.parametrize("written, found", [
    ("§ 15.50", "§ 15.50"), ("§15.50", "§15.50"), ("Tex. Bus. & Com. Code Ann. § 15.50", "§ 15.50"),
    ("Section 15.50 of the Business and Commerce Code", "Section 15.50")])
def test_statute_forms_match(written, found):
    assert citation_check.check(f"It holds under {written}.", SEARCHED)["found"] == [found]


def test_a_subsection_matches_a_stored_passage_label():
    result = citation_check.check("Courts reform it under §15.51(c), not §15.51(b).", SEARCHED)
    assert result["found"] == ["§15.51(c)"] and result["not_found"] == ["§15.51(b)"]


def test_a_reporter_citation_matches():
    result = citation_check.check("See Case A, 100 S.W. 3d 1 (Tex. App. 2019), and 93 N.Y.2d 382.", SEARCHED)
    assert result["found"] == ["100 S.W. 3d 1"] and result["not_found"] == ["93 N.Y.2d 382"]


def test_an_invented_case_is_not_found():
    result = citation_check.check("Varghese v. China Southern Airlines, 925 F.3d 1339 (11th Cir. 2019).", SEARCHED)
    assert result["not_found"] == ["925 F.3d 1339"] and not result["unchecked"]


def test_a_bare_case_name_is_unchecked():
    result = citation_check.check("As in Marsh USA Inc. v. Cook, the covenant was enforced.", SEARCHED)
    assert result == {"found": [], "not_found": [], "unchecked": ["Marsh USA Inc. v. Cook"]}


def test_several_sections_after_one_sign():
    result = citation_check.check("Under §§ 15.50 and 15.51(c), it is narrowed.", SEARCHED)
    assert result["found"] == ["§§ 15.50", "15.51(c)"]


def test_a_citation_in_the_question_counts_as_given_but_not_one_in_a_passage():
    passage = {"rows": [{"document": "d", "citation": "Agreement", "passage": "4", "text": "See § 16.70."}]}
    result = citation_check.check("Under § 15.50 and § 16.70 it is void.", [passage], "Does § 15.50 apply?")
    assert result["found"] == ["§ 15.50"] and result["not_found"] == ["§ 16.70"]


def test_a_citation_in_a_tools_error_is_not_given():
    result = citation_check.check("Under § 15.50(b) the covenant fails.", ["statutes: no document § 15.50(b)"])
    assert not result["found"] and result["not_found"] == ["§ 15.50(b)"]


def test_a_table_citation_is_resolved_in_its_step(write_environment):
    store = FakeStore()
    store.replace("laws", [{"id": i, "citation": f"Code § {i}", "title": f"Section {i}", "metadata": {},
                            "passages": [{"label": "1", "text": "x"}]} for i in ("sec-1", "sec-2")], [],
                  {"model": "toy-embed"})
    env = load_environment(write_environment(with_collection(ENV_TOML) + DOCUMENTS, dmn=cited_dmn()), store=store,
                           embedder=FakeEmbedder())
    steps = Steps(env)
    steps.decide("share_limit", share=0.5)
    assert steps.steps[0]["citations"] == [
        {"id": i, "citation": f"Code § {i}", "title": f"Section {i}", "passages": [{"label": "1", "text": "x"}],
         "collection": "laws", "built_at": None} for i in ("sec-1", "sec-2")]


def test_the_record_lists_citations_with_passages():
    calls = [{"tool": "search_cases", "arguments": {"text": "reform"}, "error": None,
              "result": SEARCHED[0] | {"built_at": "2026-11-03T12:00:00Z", "steps": [
                  {"kind": "search", "name": "cases", "inputs": {}, "result": SEARCHED[0]["rows"],
                   "built_at": "2026-11-03T12:00:00Z"}]}}]
    record = decision_record.build("Is it reasonable?", calls)
    assert record["documents"][0] == {"collection": "cases", "built_at": "2026-11-03T12:00:00Z",
                                      "document": "tx-15.51", "citation": "Tex. Bus. & Com. Code § 15.51",
                                      "passage": "(c)", "text": "The court shall reform the covenant."}
    answer = "Texas courts reform it under § 15.51(c); see Smith v. Jones; see 925 F.3d 1339."
    page = decision_record.markdown(record, answer, [], citation_check.check(answer, [SEARCHED[0]]))
    assert "## Citations\n\n- § 15.51(c): Tex. Bus. & Com. Code § 15.51 (c), cases built 2026-11-03T12:00:00Z: " \
           "\"The court shall reform the covenant.\"" in page
    assert "## Citations no tool returned\n\n925 F.3d 1339" in page
    assert "## Citations not checked (a case name without its citation)\n\nSmith v. Jones" in page
