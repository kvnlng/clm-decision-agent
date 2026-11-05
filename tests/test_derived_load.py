"""Tests for loading an environment's derived layer: which derived rules and facts are admitted, and what an
unfilled slot does. The layers are written by hand here; nothing is derived."""
# Test names say what they check.
# pylint: disable=missing-function-docstring,too-many-arguments,unbalanced-tuple-unpacking
from datetime import date
from pathlib import Path

import pytest
from fake_documents import FakeEmbedder, FakeStore

from clm_decision import derived
from clm_decision.engine import rule_info, run_decision, run_query, run_scenario
from clm_decision.environment import load_environment
from clm_decision.environment_server import describe
from clm_decision.scenario import ScenarioError

TODAY = date(2026, 12, 1)
FUNCTIONS = '''
def employee(graph, employee):
    return [graph.node(employee)]


def plan(graph):
    return [{"cap": graph.node("plan_a").get("cap")}]


def other(x):
    return {"y": x}


def branch(steps, params, levers):
    [person] = steps.query("employee", employee=params["employee"])
    limit = steps.decide("carry", years=person["years"])
    return {"value": limit["carry"]}
'''
HEAD = '''
[environment]
name = "Test"
description = "A test environment."
functions = "fx_functions"

[graph]
nodes = "nodes.csv"
edges = "edges.csv"

[documents]

[collections.hb]
description = "A handbook."
embedding_model = "m"
'''
PARTS = '''
[queries.employee]
description = "An employee."
params = { employee = "string" }
function = "employee"

[queries.plan]
description = "The plan."
params = {}
function = "plan"

[decisions.other]
description = "Another rule."
params = { x = "number" }
function = "other"

[scenarios.s]
description = "A scenario."
params = { employee = "string" }
levers = { k = [1] }
branch = "branch"
objective = "value"
limits = ["carry"]
'''
DERIVED = '''
[derived]
collection = "hb"

[derived.decisions.carry]
description = "How many unused days carry over, from full years of service."
inputs = { years = "number" }
outputs = { carry = "number", allowed = "boolean" }
owner = "People Operations"
governs = ["benefits"]

[derived.facts.plan_a.cap]
description = "The plan's cap, in dollars."
type = "number"
'''
DMN = '''<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" id="carry_definitions" name="carry">
  <decision id="carry" name="carry">
    <decisionTable id="carry_table" hitPolicy="FIRST">
      <input id="in1" label="years"><inputExpression typeRef="number"><text>years</text></inputExpression></input>
      <output id="out1" name="carry" typeRef="number"/>
      <output id="out2" name="allowed" typeRef="boolean"/>
      <output id="out3" name="reason" typeRef="string"/>
      <output id="out4" name="citation" typeRef="string"/>
      <rule id="r1"><inputEntry><text>-</text></inputEntry><outputEntry><text>7</text></outputEntry><outputEntry><text>true</text></outputEntry><outputEntry><text>"Up to 7 days."</text></outputEntry><outputEntry><text>"hb-leave"</text></outputEntry></rule>
    </decisionTable>
  </decision>
</definitions>
'''


def store() -> FakeStore:
    found = FakeStore()
    found.replace("hb", [
        {"id": "hb-leave", "citation": "§ 5", "title": "Leave", "metadata": {"effective_from": "2026-01-01"},
         "passages": [{"label": "(2)", "text": "Up to 7 unused days carry into the next year."}]},
        {"id": "hb-old", "citation": "§ 4", "title": "Old", "metadata": {"effective_from": "2024-01-01",
                                                                       "effective_until": "2025-12-31"},
         "passages": [{"label": "(4)", "text": "The cap is 5000 dollars."}]},
        {"id": "hb-plan", "citation": "§ 4", "title": "Plan", "metadata": {"effective_from": "2026-01-01"},
         "passages": [{"label": "(4)", "text": "The cap is 6300 dollars."}]}],
        [], {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "m", "dimensions": 26})
    return found


def entry(status="approved", passed=True, document="hb-leave", edition="2026-01-01") -> dict:
    return {"description": "How many unused days carry over.", "dmn": "rules/carry.dmn", "collection": "hb",
            "rule": {"statement": "Up to 7 unused days carry over.", "reason": "The leave policy.",
                     "owner": "People Operations", "version": edition, "governs": ["benefits"],
                     "cites": [document]},
            "source": {"document": document, "passages": ["(2)"], "quote": "Up to 7 unused days",
                       "edition": edition},
            "status": status, "check": None if passed is None else {"passed": passed, "failures": []},
            "decided_by": None, "decided_at": None, "note": None}


def fact(status="approved", value=6300, document="hb-plan", edition="2026-01-01") -> dict:
    return {"node": "plan_a", "property": "cap", "value": value,
            "source": {"document": document, "passages": ["(4)"], "quote": "The cap is 6300 dollars",
                       "edition": edition},
            "status": status, "check": {"passed": True, "failures": []},
            "decided_by": None, "decided_at": None, "note": None}


def make(folder: Path, *, decision=None, cap=None, plan_cap="", head=HEAD, parts=PARTS, slots=DERIVED,
         dmn=DMN) -> Path:
    (folder / "fx_functions.py").write_text(FUNCTIONS, encoding="utf-8")
    (folder / "nodes.csv").write_text(f"id,kind,name,years,cap\nplan_a,plan,Plan A,,{plan_cap}\n"
                                      "e1,employee,E1,3,\n", encoding="utf-8")
    (folder / "edges.csv").write_text("source,relation,target\n", encoding="utf-8")
    (folder / "environment.toml").write_text(head + parts + slots, encoding="utf-8")
    layer = derived.empty("hb")
    if decision is not None:
        layer["decisions"]["carry"] = decision
    if cap is not None:
        layer["facts"]["plan_a.cap"] = cap
    derived.write(folder, layer)
    (folder / "derived" / "rules").mkdir(exist_ok=True)
    (folder / "derived" / "rules" / "carry.dmn").write_text(dmn, encoding="utf-8")
    return folder / "environment.toml"


def load(path, **options):
    return load_environment(path, store=store(), embedder=FakeEmbedder(), today=TODAY, **options)


def test_without_a_derived_table_the_layer_is_ignored(tmp_path):
    env = load(make(tmp_path, decision=entry(), slots="", parts=PARTS.replace('limits = ["carry"]', "")))
    assert not env.slots and not env.unfilled and "carry" not in env.decisions


def test_an_approved_decision_loads_from_derived_rules(tmp_path):
    env = load(make(tmp_path, decision=entry()))
    assert env.decisions["carry"].origin == "derived" and env.decisions["carry"].status == "approved"
    assert "carry" in env.admitted and "carry" not in env.unfilled
    assert run_decision(env, "carry", {"years": 3})["outputs"]["carry"] == 7


def test_a_decision_whose_table_changed_since_its_check_is_not_admitted(tmp_path):
    path = make(tmp_path, decision=entry())
    layer = derived.read(tmp_path)
    layer["decisions"]["carry"]["check"]["dmn_sha256"] = derived.dmn_digest(tmp_path, layer["decisions"]["carry"])
    derived.write(tmp_path, layer)
    assert "carry" in load(path).admitted
    (tmp_path / "derived" / "rules" / "carry.dmn").write_text(DMN.replace(">7<", ">70<"), encoding="utf-8")
    env = load(path)
    assert "carry" not in env.decisions
    assert env.unfilled["carry"] == "its DMN has changed since its check: run clm-derive check again"


def test_a_checked_decision_is_not_admitted_by_default(tmp_path):
    env = load(make(tmp_path, decision=entry("checked")))
    assert "carry" not in env.decisions
    assert "checked" in env.unfilled["carry"] and "approved" in env.unfilled["carry"]


def test_admit_override_admits_proposed_and_describe_says_so(tmp_path):
    env = load(make(tmp_path, decision=entry("proposed", passed=None)), admit=["proposed", "checked", "approved"])
    assert env.admit_override and env.decisions["carry"].status == "proposed"
    shown = describe(env)["derived"]
    assert "testing" in shown["note"] and shown["slots"]["carry"]["status"] == "proposed"


def test_a_rejected_entry_is_never_admitted(tmp_path):
    with pytest.raises(ScenarioError, match="--admit-derived.*rejected"):
        load(make(tmp_path, decision=entry("rejected")), admit=["rejected"])
    with pytest.raises(ScenarioError, match="derived.admit.*rejected"):
        slots = DERIVED.replace('collection = "hb"', 'collection = "hb"\nadmit = ["rejected"]')
        load(make(tmp_path, decision=entry("rejected"), slots=slots))


def test_the_file_admits_only_approved_or_checked_and_describe_warns_of_checked(tmp_path):
    with pytest.raises(ScenarioError, match="derived.admit.*proposed.*--admit-derived"):
        load(make(tmp_path, slots=DERIVED.replace('collection = "hb"', 'collection = "hb"\nadmit = ["proposed"]')))
    env = load(make(tmp_path, decision=entry("checked"), slots=DERIVED.replace(
        'collection = "hb"', 'collection = "hb"\nadmit = ["approved", "checked"]')))
    assert env.decisions["carry"].status == "checked"
    assert "not approved by a person" in describe(env)["derived"]["note"]


def test_a_failed_check_is_never_admitted_without_override(tmp_path):
    env = load(make(tmp_path, decision=entry("approved", passed=False)))
    assert "carry" not in env.decisions and "check" in env.unfilled["carry"]


def test_a_null_check_is_not_passed(tmp_path):
    env = load(make(tmp_path, decision=entry("approved", passed=None)))
    assert "carry" not in env.decisions and "check" in env.unfilled["carry"]


def test_a_changed_edition_is_not_admitted(tmp_path):
    env = load(make(tmp_path, decision=entry(edition="2025-01-01")))
    assert "carry" not in env.decisions and "edition" in env.unfilled["carry"]


def test_an_expired_document_is_not_admitted(tmp_path):
    env = load(make(tmp_path, decision=entry(document="hb-old", edition="2024-01-01")))
    assert "carry" not in env.decisions and "not in force" in env.unfilled["carry"]


def test_a_broken_derived_dmn_leaves_the_slot_unfilled(tmp_path):
    env = load(make(tmp_path, decision=entry(), dmn="<definitions>"))
    assert "carry" not in env.decisions and "dmn" in env.unfilled["carry"]


def test_an_unfilled_limit_fails_the_branch_with_its_reason(tmp_path):
    env = load(make(tmp_path, decision=entry("checked")))
    [branch] = run_scenario(env, "s", {"employee": "e1"})["branches"]
    assert branch["status"] == "failed"
    assert "decision carry is derived and not admitted" in branch["error"] and "checked" in branch["error"]


def test_an_admitted_fact_fills_an_empty_node_property(tmp_path):
    env = load(make(tmp_path, cap=fact()))
    assert run_query(env, "plan", {})["rows"] == [{"cap": 6300}]
    assert "plan_a.cap" in env.admitted


def test_a_supplied_property_wins_over_a_derived_fact(tmp_path):
    env = load(make(tmp_path, cap=fact(value=5000), plan_cap="6300"))
    assert run_query(env, "plan", {})["rows"] == [{"cap": 6300}]
    assert "plan_a.cap" not in env.admitted and not env.derived_facts


def with_reader(text: str) -> str:
    return DERIVED.replace('collection = "hb"', f'collection = "hb"\nsecond_reader = {text}')


def test_a_second_reader_is_read_and_describe_names_it(tmp_path):
    env = load(make(tmp_path, decision=entry(), slots=with_reader('{ backend = "ollama", model = "granite4:micro" }')))
    assert env.second_reader == {"backend": "ollama", "model": "granite4:micro"}
    shown = describe(env)["derived"]
    assert shown["second_reader"] == {"backend": "ollama", "model": "granite4:micro"}
    assert "second reading by granite4:micro" in shown["second_reading"]


def test_a_second_readers_names_are_stored_stripped(tmp_path):
    env = load(make(tmp_path, decision=entry(), slots=with_reader('{ backend = "ollama", model = " qwen3:14b " }')))
    assert env.second_reader == {"backend": "ollama", "model": "qwen3:14b"}


def test_without_a_second_reader_describe_says_there_is_no_second_reading(tmp_path):
    env = load(make(tmp_path, decision=entry()))
    assert env.second_reader is None
    shown = describe(env)["derived"]
    assert shown["second_reader"] is None and "no second reading" in shown["second_reading"]


@pytest.mark.parametrize("text, wrong", [('{ backend = "openai", model = "x" }', "backend"),
                                         ('{ backend = "claude" }', "model"),
                                         ('{ backend = "claude", model = "" }', "model"),
                                         ('{ backend = "claude", model = "x", extra = 1 }', "extra"),
                                         ('"claude"', "table")])
def test_a_malformed_second_reader_is_refused_naming_the_setting(tmp_path, text, wrong):
    with pytest.raises(ScenarioError, match=f"derived.second_reader.*{wrong}"):
        load(make(tmp_path, slots=with_reader(text)))


def test_a_slot_named_like_a_supplied_decision_is_refused(tmp_path):
    with pytest.raises(ScenarioError, match="derived.decisions.other"):
        load(make(tmp_path, slots=DERIVED.replace("[derived.decisions.carry]", "[derived.decisions.other]")))


def test_fact_slots_are_refused_on_neo4j(tmp_path):
    with pytest.raises(ScenarioError, match="derived.facts.plan_a.cap.*built-in graph"):
        load(make(tmp_path), backend="neo4j")


def test_decision_steps_and_rule_info_carry_origin(tmp_path):
    env = load(make(tmp_path, decision=entry()))
    [step] = run_decision(env, "carry", {"years": 3})["steps"]
    assert step["origin"] == "derived" and step["status"] == "approved"
    assert run_decision(env, "other", {"x": 1})["steps"][0]["origin"] == "supplied"
    info = rule_info(env, "carry")
    assert info["origin"] == "derived" and info["status"] == "approved"
    assert info["source"]["document"] == "hb-leave"


def test_a_decision_result_carries_its_origin_at_the_top(tmp_path):
    env = load(make(tmp_path, decision=entry()))
    result = run_decision(env, "carry", {"years": 3})
    assert result["origin"] == "derived" and result["status"] == "approved"
    supplied = run_decision(env, "other", {"x": 1})
    assert supplied["origin"] == "supplied" and "status" not in supplied


def test_a_query_reading_a_derived_fact_carries_origin(tmp_path):
    env = load(make(tmp_path, cap=fact()))
    [step] = run_query(env, "plan", {})["steps"]
    assert step["origin"] == "derived" and step["derived"] == ["plan_a.cap"]
    assert "origin" not in run_query(env, "employee", {"employee": "e1"})["steps"][0]


def test_an_environment_with_documents_and_no_rules_loads(tmp_path):
    env = load(make(tmp_path, parts="", slots=""))
    assert not env.decisions and not env.scenarios and "hb" in env.collections


@pytest.mark.parametrize("text", ['{"version": 2}', "{bad"])
def test_a_broken_layer_is_ignored_without_a_derived_table(tmp_path, text):
    path = make(tmp_path, decision=entry(), slots="", parts=PARTS.replace('limits = ["carry"]', ""))
    derived.path(tmp_path).write_text(text, encoding="utf-8")
    assert not load(path).slots


@pytest.mark.parametrize("text", ['{"version": 2}', "{bad"])
def test_a_broken_layer_with_a_derived_table_is_one_line(tmp_path, text):
    path = make(tmp_path, decision=entry())
    derived.path(tmp_path).write_text(text, encoding="utf-8")
    with pytest.raises(ScenarioError, match=r"^derived: "):
        load(path)


def test_a_decision_derived_from_an_undated_document_is_admitted(tmp_path):
    path = make(tmp_path, decision=entry(edition=None))
    layer = derived.read(tmp_path)
    layer["decisions"]["carry"]["rule"]["version"] = ""      # what clm-derive run writes for an undated document
    derived.write(tmp_path, layer)
    undated = store()
    undated.replace("hb", [{"id": "hb-leave", "citation": "§ 5", "title": "Leave", "metadata": {},
                            "passages": [{"label": "(2)", "text": "Up to 7 unused days carry into the next year."}]}],
                    [], {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "m", "dimensions": 26})
    env = load_environment(path, store=undated, embedder=FakeEmbedder(), today=TODAY)
    assert "carry" in env.admitted, env.unfilled
    assert env.decisions["carry"].rule.version == "undated"


@pytest.mark.parametrize("side", ["inputs", "outputs"])
def test_a_derived_decision_cannot_take_or_give_a_list(tmp_path, side):
    slots = DERIVED.replace('inputs = { years = "number" }', 'inputs = { years = "list" }') if side == "inputs" \
        else DERIVED.replace('outputs = { carry = "number"', 'outputs = { carry = "list"')
    with pytest.raises(ScenarioError, match=f"^derived.decisions.carry.{side}.*: .*cannot .* a list"):
        load(make(tmp_path, slots=slots))


def test_a_supplied_property_says_why_its_derived_fact_is_not_used(tmp_path):
    env = load(make(tmp_path, cap=fact(value=5000), plan_cap="6300"))
    assert "supplied graph has a value" in env.unfilled["plan_a.cap"]
    assert "supplied graph has a value" in describe(env)["derived"]["slots"]["plan_a.cap"]["why"]


def test_a_limit_or_approval_naming_a_fact_slot_is_refused(tmp_path):
    with pytest.raises(ScenarioError, match=r"^scenarios.s.limits: 'plan_a.cap' is a derived fact, not a decision"):
        load(make(tmp_path, parts=PARTS.replace('limits = ["carry"]', 'limits = ["plan_a.cap"]')))
    with pytest.raises(ScenarioError, match=r"^scenarios.s.approvals: 'plan_a.cap' is a derived fact"):
        load(make(tmp_path, parts=PARTS + 'approvals = ["plan_a.cap"]\n'))


@pytest.mark.parametrize("admit", [["proposed"], ["rejected"]])
def test_admit_derived_without_a_derived_table_is_refused(tmp_path, admit):
    path = make(tmp_path, slots="", parts=PARTS.replace('limits = ["carry"]', ""))
    with pytest.raises(ScenarioError, match=r"^--admit-derived: "):
        load(path, admit=admit)
