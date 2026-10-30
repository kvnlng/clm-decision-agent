"""Tests for examples/new_site: a made-up bakery chain asking whether it can open a site in an expensive area."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from pathlib import Path

import pytest

from clm_decision import engine
from clm_decision.environment import load_environment
from clm_decision.environment_server import rules_for

EXAMPLE = Path(__file__).parent.parent / "examples" / "new_site" / "environment.toml"


@pytest.fixture(scope="module")
def env():
    return load_environment(EXAMPLE)


def sites(env, neighbourhood):
    result = engine.run_scenario(env, "new_site", {"neighbourhood": neighbourhood})
    return result, {branch["levers"]["size"]: branch for branch in result["branches"]}


def test_rules_for_open_site_list_what_to_ask(env):
    found = rules_for(env, "open_site")
    assert [item["name"] for item in found["to_ask"]] == ["neighbourhood"]
    names = [rule["name"] for rule in found["rules"]]
    assert set(names) == {"capital_approval", "debt_covenant", "payback", "reserve_policy", "stress_test"}
    assert names.index("payback") < names.index("stress_test")


def test_the_expensive_neighbourhood_needs_the_board_and_reports_headroom(env):
    result, by_size = sites(env, "tract_4102")
    small = by_size["small"]
    assert result["best"] == {"size": "small"} and small["status"] == "kept"
    assert small["approvals"][0]["who"] == "the board"
    # At the census estimate's low, both payback rules fail; the stress test fails first, after a fall of 1.7%
    assert [(item["case"], item["limit"]) for item in small["warnings"]] == [("low", "payback"),
                                                                             ("low", "stress_test")]
    down = next(item for item in small["headroom"] if item["direction"] == "down")
    assert down["limit"] == "stress_test" and down["change"] == pytest.approx(-0.017, abs=0.005)
    assert {item["decision"] for item in by_size["large"]["rejected_by"]} == {"debt_covenant", "stress_test"}


def test_one_neighbourhood_fails_the_stress_test(env):
    _, by_size = sites(env, "tract_3900")
    assert {branch["status"] for branch in by_size.values()} == {"rejected"}
    assert {"payback", "stress_test"} <= {item["decision"] for item in by_size["small"]["rejected_by"]}


def test_every_figure_has_a_kind(env):
    result, _ = sites(env, "tract_6400")
    steps = [step for branch in result["branches"] for step in branch["steps"] if step["kind"] == "query"]
    assert {step["information"] for step in steps} == {"fact", "estimate"}
    income = next(step for step in steps if step["name"] == "household_income")["result"][0]
    assert income["central"] == 223462 and (income["low"], income["high"]) == (223462 - 55486, 223462 + 55486)
    assert income["source"] == "ACS 2023 5-year, table B19013, Census Tract 64, King County, Washington"


def test_an_unknown_neighbourhood_fails_naming_it(env):
    result, _ = sites(env, "tract_0001")
    assert {branch["status"] for branch in result["branches"]} == {"failed"}
    assert "no neighbourhood tract_0001" in result["branches"][0]["error"]
