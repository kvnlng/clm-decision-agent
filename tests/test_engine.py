"""Tests for engine.py: queries, decisions, calculations and scenarios, with every step recorded."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
from conftest import ENV_TOML, limit_dmn

from engine import StepError, run_calculation, run_decision, run_query, run_scenario
from environment import load_environment


@pytest.fixture
def env(write_environment):
    return load_environment(write_environment())


def test_steps_record_every_number(env):
    result = run_query(env, "suppliers", {"product": "P1"})
    assert [row["cost"] for row in result["rows"]] == [10, 6]
    assert result["steps"] == [{"kind": "query", "name": "suppliers", "inputs": {"product": "P1"},
                                "result": result["rows"]}]
    decided = run_decision(env, "share_limit", {"share": 1.0})
    assert decided["outputs"]["allowed"] is False and decided["fired"] == ["too_much"]
    assert decided["steps"][0]["fired"] == ["too_much"]
    assert run_calculation(env, "cost", {"share": 0.5, "cost_a": 10, "cost_b": 6})["result"]["margin"] == 12


def test_every_branch_runs_by_default(env):
    result = run_scenario(env, "sourcing", {"product": "P1"})
    by_share = {branch["levers"]["share"]: branch for branch in result["branches"]}
    assert len(result["branches"]) == 3
    assert by_share[1.0]["status"] == "rejected"
    assert by_share[1.0]["rejected_by"] == [{"decision": "share_limit", "reason": "Too much from Region B",
                                            "inputs": {"share": 1.0}}]
    assert by_share[0.5]["status"] == by_share[0]["status"] == "kept"
    assert result["ranking"] == [{"share": 0.5}, {"share": 0}] and result["best"] == {"share": 0.5}
    assert by_share[0.5]["objective"] == 12 and result["objective"] == "margin" and result["maximise"]
    assert [step["kind"] for step in by_share[0.5]["steps"]] == ["query", "calculation", "decision"]


def test_chosen_branches_only(env):
    result = run_scenario(env, "sourcing", {"product": "P1"}, branches=[{"share": 0}])
    assert [branch["levers"] for branch in result["branches"]] == [{"share": 0}]


def test_a_lever_value_not_allowed_is_refused(env):
    with pytest.raises(StepError, match=r"share.*0\.7.*\[0, 0\.5, 1\.0\]"):
        run_scenario(env, "sourcing", {"product": "P1"}, branches=[{"share": 0.5}, {"share": 0.7}])
    with pytest.raises(StepError, match="unknown lever 'volume'"):
        run_scenario(env, "sourcing", {"product": "P1"}, branches=[{"share": 0, "volume": 2}])
    assert env.functions.CALLS == []


def test_a_missing_parameter_is_named(env):
    with pytest.raises(StepError, match="query suppliers: missing parameter product"):
        run_query(env, "suppliers", {})
    with pytest.raises(StepError, match="calculation cost: share must be a number"):
        run_calculation(env, "cost", {"share": "half", "cost_a": 10, "cost_b": 6})
    with pytest.raises(StepError, match="calculation cost: share must be a number"):
        run_calculation(env, "cost", {"share": True, "cost_a": 10, "cost_b": 6})
    with pytest.raises(StepError, match="query suppliers: unknown parameter region"):
        run_query(env, "suppliers", {"product": "P1", "region": "A"})
    with pytest.raises(StepError, match="no query 'parts'"):
        run_query(env, "parts", {})


def test_an_unchecked_limit_fails_the_branch(write_environment):
    env = load_environment(write_environment(ENV_TOML.replace('branch = "branch"', 'branch = "branch_without_limit"')))
    result = run_scenario(env, "sourcing", {"product": "P1"})
    assert {branch["status"] for branch in result["branches"]} == {"failed"}
    assert "limit share_limit was not checked" in result["branches"][0]["error"]
    assert result["ranking"] == [] and result["best"] is None


def test_a_failing_step_fails_only_its_branch(write_environment):
    env = load_environment(write_environment(dmn=limit_dmn(allowed="[0.1..0.6]")))
    result = run_scenario(env, "sourcing", {"product": "P1"})
    by_share = {branch["levers"]["share"]: branch for branch in result["branches"]}
    assert by_share[0]["status"] == "failed" and "share_limit: no rule matches share=0" in by_share[0]["error"]
    assert by_share[0.5]["status"] == "kept" and by_share[1.0]["status"] == "rejected"


def test_a_list_parameter_must_be_a_list(write_environment):
    toml = ENV_TOML.replace('params = { share = "number", cost_a', 'params = { share = "list", cost_a')
    env = load_environment(write_environment(toml))
    with pytest.raises(StepError, match="calculation cost: share must be a list"):
        run_calculation(env, "cost", {"share": 0.5, "cost_a": 10, "cost_b": 6})


def test_a_boolean_lever_can_be_chosen(write_environment):
    toml = ENV_TOML.replace("levers = { share = [0, 0.5, 1.0] }",
                            "levers = { share = [0, 0.5, 1.0], air = [true, false] }")
    env = load_environment(write_environment(toml))
    chosen = [{"share": 0, "air": True}, {"share": 1, "air": False}]
    result = run_scenario(env, "sourcing", {"product": "P1"}, branches=chosen)
    assert [branch["status"] for branch in result["branches"]] == ["kept", "rejected"]
    with pytest.raises(StepError, match="air = 1 is not allowed"):
        run_scenario(env, "sourcing", {"product": "P1"}, branches=[{"share": 0, "air": 1}])


def test_a_swallowed_limit_failure_fails_the_branch(write_environment):
    toml = ENV_TOML.replace('branch = "branch"', 'branch = "branch_swallowing"')
    env = load_environment(write_environment(toml, dmn=limit_dmn(allowed="[0.1..0.6]")))
    half = run_scenario(env, "sourcing", {"product": "P1"}, branches=[{"share": 0.5}])["branches"][0]
    assert half["status"] == "failed" and "limit share_limit failed: share_limit: no rule matches" in half["error"]
    assert any(step.get("error") for step in half["steps"])
