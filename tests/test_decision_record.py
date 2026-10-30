"""Tests for decision_record.py: the officer's record, built only from the tools' results."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json

import pytest
from test_risk import RISK_FUNCTIONS, RISK_TOML
from test_rules import RULES_TOML, one_input_dmn

from clm_decision import decision_record, engine
from clm_decision.environment import load_environment
from clm_decision.environment_server import rules_for


def call(tool, arguments, result=None, error=None):
    """A call as clm-agent records it, with the result as it arrives over MCP (JSON)."""
    return {"tool": tool, "arguments": arguments, "result": json.loads(json.dumps(result)), "error": error}


@pytest.fixture
def calls(write_environment):
    env = load_environment(write_environment(RISK_TOML, RISK_FUNCTIONS))
    return [call("describe", {}, {"name": "Toy sourcing"}),
            call("scenario_risky", {"product": "P1"}, engine.run_scenario(env, "risky", {"product": "P1"}))]


def test_the_record_is_built_from_calls_only(calls):
    assert decision_record.build("Should we?", calls) == decision_record.build("Should we?", calls)
    record = decision_record.build("Should we, for P1?", calls)
    assert record["question"] == "Should we, for P1?" and record["supplied"] == {"product": ["P1"]}
    scenario = record["scenarios"][0]
    assert scenario["scenario"] == "risky" and scenario["best"] == {"share": 0.5}
    assert {branch["status"] for branch in scenario["branches"]} == {"kept"}
    assert "steps" not in scenario["branches"][0]


def test_the_record_lists_estimates_with_range_and_source(calls):
    estimates = [item for item in decision_record.build("Q", calls)["information"] if item["kind"] == "estimate"]
    assert estimates == [{"query": "demand", "kind": "estimate", "inputs": {"product": "P1"},
                          "description": "Units the product may sell: low, central and high",
                          "rows": [{"low": 80, "central": 100, "high": 130, "source": "survey", "as_of": "2025"}]}]


def test_the_record_shows_what_is_still_to_ask(write_environment):
    path = write_environment(RULES_TOML)
    (path.parent / "rules" / "payback.dmn").write_text(one_input_dmn("payback", "payback_years", "&gt; 4"))
    (path.parent / "rules" / "reserve_policy.dmn").write_text(
        one_input_dmn("reserve_policy", "share_of_reserves", "&gt; 0.5"))
    found = rules_for(load_environment(path), "open_site")
    record = decision_record.build("Can we open a site?", [call("rules", {"topic": "open_site"}, found)])
    assert [item["name"] for item in record["to_ask"]] == ["site_size"]
    assert set(record["rules_consulted"]) == {"payback", "reserve_policy"}


def test_the_record_says_when_no_scenario_ran():
    record = decision_record.build("Q", [call("scenario_risky", {"product": "P9"}, error="unknown product")])
    assert not record["scenarios"] and record["failed_calls"] == [{"tool": "scenario_risky",
                                                                     "error": "unknown product"}]
    assert "No scenario was run." in decision_record.markdown(record, "", [])


def test_markdown_marks_the_agents_explanation_as_unchecked(calls):
    text = decision_record.markdown(decision_record.build("Should we?", calls), "Buy half from B.", ["42"])
    assert "## The agent's explanation (not checked)" in text and "Buy half from B." in text
    assert text.index("## Rules applied") < text.index("## The agent's explanation")
    assert "board_approval" in text and "the board" in text
    assert "demand: down -25" in text.replace(".0%", "%")
    assert "42" in text.split("## Figures no tool returned")[1]


def test_the_record_shows_each_facts_values(calls):
    facts = [item for item in decision_record.build("Q", calls)["information"] if item["kind"] == "fact"]
    assert facts[0]["query"] == "suppliers" and facts[0]["rows"][0] == {"id": "S1", "region": "A", "cost": 10}
    assert '"cost": 10' in decision_record.markdown(decision_record.build("Q", calls), "", [])


def test_a_value_not_in_the_question_is_the_agents_choice_not_supplied(calls):
    record = decision_record.build("Should we buy more from Region B?", calls)
    assert not record["supplied"] and record["chosen_by_agent"] == {"product": ["P1"]}
    assert "## Chosen by the agent, not in the question" in decision_record.markdown(record, "", [])


def test_a_value_in_the_question_is_supplied(calls):
    assert decision_record.build("Should we, for product P1?", calls)["supplied"] == {"product": ["P1"]}


def test_a_missing_item_the_agent_filled_in_stays_to_ask(write_rules_env):
    env, rules_result = write_rules_env
    calls = [call("rules", {"topic": "open_site"}, rules_result),
             call("scenario_x", {}, {"scenario": "x", "objective": "o", "maximise": True, "best": None,
                                     "ranking": [], "branches": [], "supplied": {"site_size": "large"}})]
    record = decision_record.build("Can we open a site?", calls)
    assert [item["name"] for item in record["to_ask"]] == ["site_size"] and env


def test_each_scenario_run_shows_its_own_parameters(write_environment):
    env = load_environment(write_environment(RISK_TOML, RISK_FUNCTIONS))
    calls = [call("scenario_risky", {"product": product}, engine.run_scenario(env, "risky", {"product": product}))
             for product in ("P1", "P1")]
    calls[1]["result"]["supplied"] = {"product": "P2"}
    record = decision_record.build("Compare P1 and P2", calls)
    assert [scenario["supplied"] for scenario in record["scenarios"]] == [{"product": "P1"}, {"product": "P2"}]
    assert record["supplied"] == {"product": ["P1", "P2"]}
    assert "**risky** (product = P2)" in decision_record.markdown(record, "", [])


def test_rules_only_looked_at_are_consulted_not_applied(write_rules_env):
    _, rules_result = write_rules_env
    record = decision_record.build("Q", [call("rules", {"topic": "open_site"}, rules_result)])
    assert not record["rules"] and set(record["rules_consulted"]) == {"payback", "reserve_policy"}
    text = decision_record.markdown(record, "", [])
    assert "## Rules consulted" in text and "No rule was applied" in text


@pytest.fixture
def write_rules_env(write_environment):
    path = write_environment(RULES_TOML)
    (path.parent / "rules" / "payback.dmn").write_text(one_input_dmn("payback", "payback_years", "&gt; 4"))
    (path.parent / "rules" / "reserve_policy.dmn").write_text(
        one_input_dmn("reserve_policy", "share_of_reserves", "&gt; 0.5"))
    env = load_environment(path)
    return env, rules_for(env, "open_site")
