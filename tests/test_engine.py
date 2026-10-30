"""Tests for engine.py: queries, decisions, calculations and scenarios, with every step recorded."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import threading

import pytest
from conftest import ENV_FUNCTIONS, ENV_TOML, JUDGE_TOML, limit_dmn

from clm_decision.clm_common import ClmError
from clm_decision.engine import StepError, run_calculation, run_decision, run_judgement, run_query, run_scenario
from clm_decision.environment import load_environment
from clm_decision.scenario import ScenarioError


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


def test_no_branches_and_a_repeated_branch_are_refused(env):
    with pytest.raises(StepError, match="scenario sourcing: choose at least one branch, or leave branches out to run "
                                        "them all"):
        run_scenario(env, "sourcing", {"product": "P1"}, branches=[])
    with pytest.raises(StepError, match=r"scenario sourcing: the branch \{'share': 0\.5\} is given twice"):
        run_scenario(env, "sourcing", {"product": "P1"}, branches=[{"share": 0.5}, {"share": 0}, {"share": 0.5}])
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


def test_a_function_decision_must_return_a_dict(write_environment):
    toml = ENV_TOML + '\n[decisions.listed]\ndescription = "Answers with a list"\nparams = {}\nfunction = "listed"\n'
    env = load_environment(write_environment(toml, ENV_FUNCTIONS + "\n\ndef listed():\n    return [True]\n"))
    with pytest.raises(StepError, match="decision listed: its function must return a dict"):
        run_decision(env, "listed", {})


@pytest.mark.parametrize("value", ['float("nan")', 'float("inf")'])
def test_an_objective_that_is_not_finite_fails_the_branch(write_environment, value):
    functions = ENV_FUNCTIONS + f"\n\ndef branch_odd(steps, params, levers):\n    return {{**branch(steps, " \
                                f"params, levers), 'margin': {value}}}\n"
    env = load_environment(write_environment(ENV_TOML.replace('branch = "branch"', 'branch = "branch_odd"'), functions))
    result = run_scenario(env, "sourcing", {"product": "P1"}, branches=[{"share": 0}])
    assert result["branches"][0]["status"] == "failed" and result["best"] is None
    assert "scenario sourcing: the figures' margin is not a finite number" in result["branches"][0]["error"]


def test_a_swallowed_limit_failure_fails_the_branch(write_environment):
    toml = ENV_TOML.replace('branch = "branch"', 'branch = "branch_swallowing"')
    env = load_environment(write_environment(toml, dmn=limit_dmn(allowed="[0.1..0.6]")))
    half = run_scenario(env, "sourcing", {"product": "P1"}, branches=[{"share": 0.5}])["branches"][0]
    assert half["status"] == "failed" and "limit share_limit failed: share_limit: no rule matches" in half["error"]
    assert any(step.get("error") for step in half["steps"])


def test_a_judge_needs_clm(write_environment):
    toml = ENV_TOML + JUDGE_TOML.replace('[clm]\nbackend = "mlx"\n', "")
    with pytest.raises(ScenarioError, match="^judges.risk: a judge needs CLM: add a \\[clm\\] table"):
        load_environment(write_environment(toml))


def test_judgement_is_traced(write_environment, fake_clm):
    env = load_environment(write_environment(ENV_TOML + JUDGE_TOML))
    assert fake_clm.loads == 0                  # not at load: only when something is judged
    result = run_judgement(env, "risk", {"text": "Workers vote to strike at the plant"})
    assert result["result"] == {"probability": 0.9}
    assert result["steps"] == [{"kind": "judgement", "name": "risk", "inputs": {"text": "Workers vote to strike at "
                                "the plant"}, "result": {"probability": 0.9}}]
    question = fake_clm.asked[0][1]
    assert list(question.values()) == [{"type": "noul", "instructions": "Does this news mean the supplier may fail "
                                                                        "to deliver?"}]


def test_clm_loads_once_on_first_judgement(write_environment, fake_clm):
    env = load_environment(write_environment(ENV_TOML + JUDGE_TOML))
    threads = [threading.Thread(target=run_judgement, args=(env, "risk", {"text": f"news {n}"})) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert fake_clm.loads == 1 and len(fake_clm.asked) == 4


def test_empty_text_is_refused(write_environment, fake_clm):
    env = load_environment(write_environment(ENV_TOML + JUDGE_TOML))
    with pytest.raises(StepError, match="judge risk: there is no text to judge"):
        run_judgement(env, "risk", {"text": "  "})
    assert fake_clm.asked == []


def test_a_clm_that_cannot_load_fails_the_branch(write_environment, monkeypatch):
    from clm_decision import clm_router  # pylint: disable=import-outside-toplevel

    def broken(*args, **kwargs):
        raise ClmError("The mlx backend needs an Apple Silicon Mac.")
    monkeypatch.setattr(clm_router, "load_engine", broken)
    functions = ENV_FUNCTIONS + '''

def judging_branch(steps, params, levers):
    figures = branch(steps, params, levers)
    steps.judge("risk", text="Workers vote to strike")
    return figures
'''
    env = load_environment(write_environment(ENV_TOML.replace('branch = "branch"', 'branch = "judging_branch"')
                                             + JUDGE_TOML, functions=functions))
    branch = run_scenario(env, "sourcing", {"product": "P1"}, branches=[{"share": 0}])["branches"][0]
    assert branch["status"] == "failed" and branch["error"].startswith("judge risk: CLM could not load: The mlx")


def test_a_multiline_load_error_is_one_line(write_environment, monkeypatch):
    from clm_decision import clm_router  # pylint: disable=import-outside-toplevel

    def missing(*args, **kwargs):
        raise RuntimeError("404 Client Error.\nRepository Not Found for url: https://huggingface.co/x\nCheck the name.")
    monkeypatch.setattr(clm_router, "load_engine", missing)
    env = load_environment(write_environment(ENV_TOML + JUDGE_TOML))
    with pytest.raises(StepError) as caught:
        run_judgement(env, "risk", {"text": "a strike is called"})
    assert str(caught.value) == "judge risk: CLM could not load: 404 Client Error."


def test_clm_heads_is_refused_for_judges(write_environment):
    toml = ENV_TOML + JUDGE_TOML.replace('backend = "mlx"\n', 'backend = "mlx"\nheads = "heads"\n')
    with pytest.raises(ScenarioError, match="^clm.heads: not supported for judges yet"):
        load_environment(write_environment(toml))


def test_a_judge_without_the_clm_extra_says_how_to_install_it(write_environment, monkeypatch):
    import sys  # pylint: disable=import-outside-toplevel
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.delitem(sys.modules, "clm_decision.clm_router")
    monkeypatch.delattr("clm_decision.clm_router")             # so that `from . import clm_router` imports it again
    env = load_environment(write_environment(ENV_TOML + JUDGE_TOML))
    with pytest.raises(StepError, match=r"^judge risk: CLM could not load: CLM is not installed \(huggingface_hub"):
        run_judgement(env, "risk", {"text": "a strike is called"})
