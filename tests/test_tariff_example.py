"""Tests for the tariff example: the COO's sourcing question, decided from the graph, the rules and the levers."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from pathlib import Path

import pytest

from engine import run_query, run_scenario
from environment import load_environment

EXAMPLE = Path(__file__).parent.parent / "examples" / "tariff" / "environment.toml"
LIMITS = ("sla_threshold", "quality_threshold", "supply_concentration")


@pytest.fixture(scope="module")
def env():
    return load_environment(EXAMPLE)


@pytest.fixture(scope="module")
def result(env):
    return run_scenario(env, "sourcing", {"product": "PX"})


def branch(result, share):
    return next(item for item in result["branches"] if item["levers"] == {"region_b_share": share})


def test_moving_everything_is_rejected_by_the_shared_provider(result):
    everything = branch(result, 1.0)
    assert everything["status"] == "rejected"
    assert [item["decision"] for item in everything["rejected_by"]] == ["supply_concentration"]
    assert everything["rejected_by"][0]["inputs"]["provider"] == "Kestrel Polymers"


def test_half_is_best(result):
    assert result["best"] == {"region_b_share": 0.5}
    assert result["ranking"] == [{"region_b_share": 0.5}, {"region_b_share": 0}]
    # Region B is cheaper once tariffs are paid, so moving everything would have earned the most
    assert branch(result, 1.0)["objective"] > branch(result, 0.5)["objective"] > branch(result, 0)["objective"]


def test_every_branch_checks_every_limit(env, result):
    clients = run_query(env, "key_clients_of", {"product": "PX"})["rows"]
    assert len(clients) >= 2
    for item in result["branches"]:
        decided = [step["name"] for step in item["steps"] if step["kind"] == "decision"]
        assert all(limit in decided for limit in LIMITS)
        assert decided.count("sla_threshold") == len(clients)


def _numbers(value):
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]
    if isinstance(value, list):
        return [number for item in value for number in _numbers(item)]
    return [value] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


def test_every_figure_comes_from_a_step(result):
    kept = branch(result, 0.5)
    found = [number for step in kept["steps"] for number in _numbers(step["result"])]
    assert kept["figures"] and all(value in found for value in _numbers(kept["figures"]))


def test_the_dependency_is_found_by_query(env):
    region_b = [row["id"] for component in run_query(env, "components_of", {"product": "PX"})["rows"]
                for row in run_query(env, "suppliers_of", {"component": component["id"]})["rows"]
                if row["region"] == "Region B"]
    shared = run_query(env, "single_points_of_failure", {"suppliers": sorted(set(region_b))})["rows"]
    assert [row["name"] for row in shared] == ["Kestrel Polymers"]
