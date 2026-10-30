"""Tests for kinds of information in the engine: a fact as recorded, an estimate with its range, source and date."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
from conftest import ENV_FUNCTIONS, ENV_TOML

from clm_decision import engine
from clm_decision.environment import load_environment
from clm_decision.scenario import StepError

ESTIMATE_TOML = ENV_TOML + '''
[queries.demand]
description = "Units a product may sell next quarter: low, central and high, with source and date"
params = { product = "string" }
function = "demand"
kind = "estimate"

[decisions.share_limit.rule]
statement = "At most 90% from Region B."
reason = "One region's disruption must not stop production."
owner = "Supply chain"
version = "2026-01"
governs = ["choose_supplier"]
'''

ESTIMATE_FUNCTIONS = ENV_FUNCTIONS + '''
DEMAND = {"low": 80, "central": 100, "high": 130, "source": "Retail survey, table 4", "as_of": "2025"}


def demand(graph, product):
    return [{"product": product, **DEMAND}]
'''


@pytest.fixture
def env(write_environment):
    return load_environment(write_environment(ESTIMATE_TOML, ESTIMATE_FUNCTIONS))


def test_an_estimate_row_gets_its_central_value(env):
    row = engine.Steps(env).query("demand", product="P1")[0]
    assert row["value"] == 100 and (row["low"], row["high"]) == (80, 130) and row["source"] == "Retail survey, table 4"


@pytest.mark.parametrize("case, value", [("low", 80), ("high", 130), (0.9, 90.0), ("central", 100)])
def test_a_case_picks_low_or_high_or_a_multiple(env, case, value):
    steps = engine.Steps(env, {"demand": case})
    assert steps.query("demand", product="P1")[0]["value"] == pytest.approx(value)
    assert steps.steps[-1]["case"] == case


@pytest.mark.parametrize("missing", ["low", "central", "high", "source", "as_of"])
def test_an_estimate_row_without_its_range_is_one_line(env, missing):
    del env.functions.DEMAND[missing]
    with pytest.raises(StepError, match="^query demand: an estimate's rows need low, central, high, source and as_of"):
        engine.Steps(env).query("demand", product="P1")


def test_low_above_central_is_refused(env):
    env.functions.DEMAND["low"] = 120
    with pytest.raises(StepError, match=r"^query demand: an estimate's low <= central <= high, not 120, 100, 130"):
        engine.Steps(env).query("demand", product="P1")


def test_steps_say_what_kind_of_information(env):
    steps = engine.Steps(env)
    steps.query("suppliers", product="P1")
    steps.query("demand", product="P1")
    assert [step["information"] for step in steps.steps] == ["fact", "estimate"]
    assert "case" not in steps.steps[0] and steps.steps[1]["case"] == "central"


def test_a_scenario_result_names_its_rules_information_and_supplied_params(env):
    result = engine.run_scenario(env, "sourcing", {"product": "P1"})
    assert result["supplied"] == {"product": "P1"}
    assert result["rules"]["share_limit"]["owner"] == "Supply chain"
    assert result["information"] == {"suppliers": {"kind": "fact",
                                                   "description": "The product's suppliers, with region and unit cost"}}


def test_a_fact_query_is_unchanged(env):
    rows = engine.Steps(env).query("suppliers", product="P1")
    assert all("value" not in row for row in rows)
