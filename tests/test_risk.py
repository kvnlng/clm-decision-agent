"""Tests for risk in scenarios: warnings at an estimate's low and high, headroom, and approvals."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
from conftest import ENV_FUNCTIONS, ENV_TOML

from clm_decision import engine
from clm_decision.environment import load_environment
from clm_decision.scenario import ScenarioError

RISK_TOML = ENV_TOML + '''
[queries.demand]
description = "Units the product may sell: low, central and high"
params = { product = "string" }
function = "demand"
kind = "estimate"

[decisions.margin_floor]
description = "Whether the profit is at least 900"
params = { profit = "number" }
function = "margin_floor"

[decisions.board_approval]
description = "Whether the board must approve"
params = { profit = "number" }
function = "board"

[calculations.profit]
description = "Profit from units and the margin per unit"
params = { units = "number", margin = "number" }
function = "profit"

[scenarios.risky]
description = "How much to buy from Region B, with uncertain demand"
params = { product = "string" }
levers = { share = [0, 0.5] }
branch = "risky_branch"
objective = "profit"
limits = ["share_limit", "margin_floor"]
uncertain = ["demand"]
approvals = ["board_approval"]

[scenarios.plain_uncertain]
description = "The sourcing scenario, declaring an estimate it never uses"
params = { product = "string" }
levers = { share = [0, 0.5] }
branch = "branch"
objective = "margin"
limits = ["share_limit"]
uncertain = ["demand"]

[scenarios.unchecked_approval]
description = "Declares an approval its branch never checks"
params = { product = "string" }
levers = { share = [0] }
branch = "branch"
objective = "margin"
limits = ["share_limit"]
approvals = ["board_approval"]
'''

RISK_FUNCTIONS = ENV_FUNCTIONS + '''

def demand(graph, product):
    return [{"product": product, "low": 80, "central": 100, "high": 130, "source": "survey", "as_of": "2025"}]


def margin_floor(profit):
    return {"allowed": profit >= 900, "reason": "Profit below 900" if profit < 900 else "Profit is enough"}


def board(profit):
    return {"required": profit > 1100, "who": "the board", "reason": "Large projects go to the board"}


def profit(units, margin):
    return {"profit": units * margin}


def risky_branch(steps, params, levers):
    figures = branch(steps, params, levers)
    units = steps.query("demand", product=params["product"])[0]["value"]
    result = steps.calculate("profit", units=units, margin=figures["margin"])
    steps.decide("margin_floor", profit=result["profit"])
    steps.decide("board_approval", profit=result["profit"])
    return result
'''


@pytest.fixture
def env(write_environment):
    return load_environment(write_environment(RISK_TOML, RISK_FUNCTIONS))


def branches(result):
    return {branch["levers"]["share"]: branch for branch in result["branches"]}


def test_a_branch_failing_only_at_the_low_estimate_is_kept_with_a_warning(env):
    # share 0: margin 10 a unit, so profit 1000 at central demand but 800 at the low estimate
    zero = branches(engine.run_scenario(env, "risky", {"product": "P1"}))[0]
    assert zero["status"] == "kept"
    assert zero["warnings"] == [{"estimate": "demand", "case": "low", "limit": "margin_floor",
                                 "reason": "Profit below 900"}]


def test_headroom_finds_how_far_the_estimate_can_fall(env):
    # share 0.5: margin 12 a unit, profit 1200; it falls below 900 when demand falls by a quarter
    half = branches(engine.run_scenario(env, "risky", {"product": "P1"}))[0.5]
    down = next(item for item in half["headroom"] if item["direction"] == "down")
    assert down["estimate"] == "demand" and down["limit"] == "margin_floor"
    assert down["change"] == pytest.approx(-0.25, abs=0.005)


def test_headroom_is_none_when_no_limit_fails_in_range(env):
    half = branches(engine.run_scenario(env, "risky", {"product": "P1"}))[0.5]
    assert next(item for item in half["headroom"] if item["direction"] == "up") == {
        "estimate": "demand", "direction": "up", "change": None, "limit": None}


def test_a_branch_not_using_the_estimate_has_no_warnings(env):
    for branch in engine.run_scenario(env, "plain_uncertain", {"product": "P1"})["branches"]:
        assert branch["warnings"] == [] and branch["headroom"] == []


def test_a_required_approval_is_listed_not_rejected(env):
    found = branches(engine.run_scenario(env, "risky", {"product": "P1"}))
    assert found[0.5]["status"] == "kept"
    assert found[0.5]["approvals"] == [{"decision": "board_approval", "who": "the board",
                                        "reason": "Large projects go to the board"}]
    assert found[0]["approvals"] == []


def test_an_unchecked_approval_fails_the_branch(env):
    branch = engine.run_scenario(env, "unchecked_approval", {"product": "P1"})["branches"][0]
    assert branch["status"] == "failed" and branch["error"] == "approval board_approval was not checked"


def test_ranking_uses_the_central_case(env):
    assert engine.run_scenario(env, "risky", {"product": "P1"})["ranking"] == [{"share": 0.5}, {"share": 0}]


@pytest.mark.parametrize("old, new, setting", [
    ('uncertain = ["demand"]\napprovals', 'uncertain = ["suppliers"]\napprovals', "scenarios.risky.uncertain"),
    ('approvals = ["board_approval"]\n\n[scenarios.plain', 'approvals = ["board"]\n\n[scenarios.plain',
     "scenarios.risky.approvals"),
])
def test_uncertain_and_approvals_must_name_the_right_parts(write_environment, old, new, setting):
    assert old in RISK_TOML
    with pytest.raises(ScenarioError, match=f"^{setting}"):
        load_environment(write_environment(RISK_TOML.replace(old, new, 1), RISK_FUNCTIONS))
