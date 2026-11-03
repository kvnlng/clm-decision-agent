"""Tests for examples/noncompete/fetch_cases.py: which opinions are kept, how they become documents, and how facts
read by a model are checked against the opinion's own words. The opinions here are made up."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json
import sys
from pathlib import Path

from clm_decision.documents import Collection
from clm_decision.load_documents import read_jsonl

sys.path.append(str(Path(__file__).parent.parent / "examples" / "noncompete"))
import fetch_cases  # noqa: E402  pylint: disable=wrong-import-position,import-error

FILTERS = {"state": "string", "court": "string", "court_level": "string", "year": "integer", "decided": "date",
           "role": "string", "length_months": "number", "area": "string", "outcome": "string",
           "reformed_to": "string"}


def case(ident=1, text="", court="Texas Courts of Appeals", court_abbreviation="Tex. App.", jurisdiction="Tex.",
         cites=()):
    return {"id": ident, "name_abbreviation": f"Made-up Co. v. Person {ident}", "decision_date": "2015-03-02",
            "citations": [{"type": "official", "cite": f"{ident} S.W.3d 10"}],
            "court": {"name": court, "name_abbreviation": court_abbreviation},
            "jurisdiction": {"name": jurisdiction},
            "cites_to": [{"cite": cite, "case_ids": [target]} for cite, target in cites],
            "casebody": {"opinions": [{"text": text}]}}


NON_COMPETE = case(1, "The sales manager signed a covenant not to compete for two years across the United States."
                      "\n\nWe hold the covenant unreasonable and reform it to the three states he served, citing "
                      "2 S.W.3d 10.", cites=[("2 S.W.3d 10", 2)])
UNRELATED = case(3, "The deed's restrictive covenant limits the lot to one house.")


def test_a_non_compete_opinion_is_kept():
    assert fetch_cases.is_non_compete(fetch_cases.opinion_text(NON_COMPETE))
    assert fetch_cases.state_of(NON_COMPETE) == "TX"


def test_an_unrelated_opinion_is_dropped():
    assert not fetch_cases.is_non_compete(fetch_cases.opinion_text(UNRELATED))
    assert fetch_cases.state_of(case(jurisdiction="Ohio")) is None


def test_a_federal_opinion_takes_the_state_whose_law_it_applies():
    fifth = case(court="United States Court of Appeals for the Fifth Circuit", court_abbreviation="5th Cir.",
                 jurisdiction="U.S.", text="Applying Texas law, we consider the covenant not to compete.")
    assert fetch_cases.state_of(fifth) == "TX" and fetch_cases.court_level(fifth) == "federal"
    assert fetch_cases.state_of(case(court="United States Court of Appeals for the Fifth Circuit",
                                     jurisdiction="U.S.", text="Under Louisiana law ...")) is None


def test_court_levels():
    assert fetch_cases.court_level(case(court="Supreme Court of Texas")) == "supreme"
    assert fetch_cases.court_level(case()) == "appeals"


def test_a_document_has_passages_metadata_and_citing_passages():
    document = fetch_cases.document(NON_COMPETE, {"role": {"value": "sales manager", "passage": "1",
                                                           "quote": "The sales manager"}})
    assert document["id"] == "cap-1" and document["citation"] == "1 S.W.3d 10"
    assert document["metadata"] | {} == {"state": "TX", "court": "Tex. App.", "court_level": "appeals",
                                         "year": 2015, "decided": "2015-03-02", "role": "sales manager",
                                         "sources": {"role": "1"}}
    assert document["cites"] == [{"id": "cap-2", "passage": "2"}]


def test_a_fact_without_its_quote_in_the_passage_is_left_empty():
    passages = [{"label": "1", "text": "The sales manager signed a covenant not to compete for two years."}]
    facts = fetch_cases.checked_facts({
        "role": {"value": "sales manager", "passage": "1", "quote": "The sales manager"},
        "length_months": {"value": 24, "passage": "1", "quote": "for two years"},
        "area": {"value": "nationwide", "passage": "1", "quote": "across the whole nation"},
        "outcome": {"value": "enforced", "passage": "9", "quote": "enforced"},
        "court_level": {"value": "supreme", "passage": "1", "quote": "The sales manager"}}, passages)
    assert facts == {"role": {"value": "sales manager", "passage": "1", "quote": "The sales manager"},
                     "length_months": {"value": 24, "passage": "1", "quote": "for two years"}}


def test_a_fact_of_the_wrong_kind_is_left_empty():
    passages = [{"label": "1", "text": "It was reformed for two years."}]
    facts = fetch_cases.checked_facts({"outcome": {"value": "maybe", "passage": "1", "quote": "reformed"},
                                       "length_months": {"value": "two", "passage": "1", "quote": "two years"}},
                                      passages)
    assert facts == {}


def test_the_output_loads(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(fetch_cases.document(NON_COMPETE, {})) + "\n")
    [loaded] = read_jsonl(path, Collection("cases", "x", FILTERS, "m"))
    assert loaded["metadata"]["state"] == "TX"
