"""Tests for rules that cite their source: a rule's `cites`, and a decision table's `citation` column, checked
against the decision's collection when the environment loads."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
from conftest import ENV_TOML
from fake_documents import FakeEmbedder, FakeStore

from clm_decision.engine import run_decision
from clm_decision.environment import load_environment
from clm_decision.environment_server import explain_rule
from clm_decision.scenario import ScenarioError

DOCUMENTS = '''
[documents]

[collections.laws]
description = "Statutes"
embedding_model = "toy-embed"
'''
RULE = '''
[decisions.share_limit.rule]
statement = "No more than 90% of a product's parts come from Region B."
reason = "Production must survive one region's disruption."
owner = "Sourcing"
version = "2026-01"
cites = ["sec-1"]
'''


def cited_dmn(citations=("sec-1", "sec-1; sec-2")) -> str:
    """The share_limit table with a citation column."""
    rule = ('<rule id="{id}"><inputEntry><text>{test}</text></inputEntry><outputEntry><text>{allowed}</text>'
            '</outputEntry><outputEntry><text>"{citation}"</text></outputEntry></rule>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" id="d" name="d" namespace="t">'
            '<decision id="share_limit" name="share_limit"><decisionTable id="t">'
            '<input id="i"><inputExpression typeRef="number"><text>share</text></inputExpression></input>'
            '<output id="o1" name="allowed" typeRef="boolean"/><output id="o2" name="citation" typeRef="string"/>'
            + rule.format(id="too_much", test="&gt; 0.9", allowed="false", citation=citations[0])
            + rule.format(id="fine", test="&lt;= 0.9", allowed="true", citation=citations[1])
            + "</decisionTable></decision></definitions>")


def store(*ids):
    found = FakeStore()
    found.replace("laws", [{"id": i, "citation": i, "title": i, "metadata": {},
                            "passages": [{"label": "1", "text": "x"}]} for i in ids], [], {"model": "toy-embed"})
    return found


def with_collection(toml: str) -> str:
    return toml.replace('dmn = "rules/share_limit.dmn"', 'dmn = "rules/share_limit.dmn"\ncollection = "laws"')


def load(write_environment, toml, ids=("sec-1", "sec-2"), dmn=None, folder="toy"):
    return load_environment(write_environment(toml, dmn=dmn, folder=folder), store=store(*ids),
                            embedder=FakeEmbedder())


def test_a_cited_rule_loads(write_environment):
    env = load(write_environment, with_collection(ENV_TOML) + DOCUMENTS + RULE)
    assert env.decisions["share_limit"].rule.cites == ("sec-1",)
    assert env.decisions["share_limit"].collection == "laws"


def test_a_rule_citation_not_in_the_collection_stops_loading(write_environment):
    with pytest.raises(ScenarioError, match=r"^decisions\.share_limit\.rule\.cites: laws holds no document sec-1$"):
        load(write_environment, with_collection(ENV_TOML) + DOCUMENTS + RULE, ids=("sec-2",))


def test_citations_need_the_decision_to_name_its_collection(write_environment):
    with pytest.raises(ScenarioError, match=r"^decisions\.share_limit\.collection: name the collection "
                                            r"its citations are in"):
        load(write_environment, ENV_TOML + DOCUMENTS + RULE)


def test_an_unknown_collection_stops_loading(write_environment):
    toml = with_collection(ENV_TOML).replace('collection = "laws"', 'collection = "nope"') + DOCUMENTS
    with pytest.raises(ScenarioError, match=r"^decisions\.share_limit\.collection: no collection nope \(laws\)$"):
        load(write_environment, toml)


def test_dmn_citation_cells_are_checked(write_environment):
    toml = with_collection(ENV_TOML) + DOCUMENTS
    load(write_environment, toml, dmn=cited_dmn())
    with pytest.raises(ScenarioError, match=r"^decisions\.share_limit\.dmn: row fine cites sec-2, "
                                            r"which laws does not hold$"):
        load(write_environment, toml, ids=("sec-1",), dmn=cited_dmn(), folder="again")


def test_rule_shows_its_citations(write_environment):
    env = load(write_environment, with_collection(ENV_TOML) + DOCUMENTS + RULE)
    shown = explain_rule(env, "share_limit")
    assert shown["cites"] == ["sec-1"] and shown["collection"] == "laws"


def test_cited_text_false_keeps_the_citation_and_drops_the_passages(write_environment):
    toml = with_collection(ENV_TOML) + DOCUMENTS.replace("[documents]", "[documents]\ncited_text = false")
    env = load(write_environment, toml, dmn=cited_dmn())
    cited = run_decision(env, "share_limit", {"share": 0.5})["steps"][0]["citations"]
    assert [c["citation"] for c in cited] == ["sec-1", "sec-2"]
    assert all("passages" not in c for c in cited)


def test_a_collect_table_cites_each_row_that_fired(write_environment):
    dmn = cited_dmn().replace('<decisionTable id="t">', '<decisionTable id="t" hitPolicy="COLLECT">') \
        .replace("&gt; 0.9", "&gt;= 0")
    env = load(write_environment, with_collection(ENV_TOML) + DOCUMENTS, dmn=dmn)
    step = run_decision(env, "share_limit", {"share": 0.5})["steps"][0]
    assert step["fired"] == ["too_much", "fine"] and [c["id"] for c in step["citations"]] == ["sec-1", "sec-2"]


def test_cited_text_must_be_true_or_false(write_environment):
    toml = with_collection(ENV_TOML) + DOCUMENTS.replace("[documents]", '[documents]\ncited_text = "no"')
    with pytest.raises(ScenarioError, match="documents.cited_text"):
        load(write_environment, toml, dmn=cited_dmn())
