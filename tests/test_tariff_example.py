"""Tests for the tariff example: the COO's sourcing question, decided from the graph, the rules and the levers."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from pathlib import Path

import pytest

from clm_decision.engine import run_query, run_scenario
from clm_decision.environment import load_environment

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


def test_sourcing_never_loads_clm(fake_clm):
    fresh = load_environment(EXAMPLE)
    assert run_scenario(fresh, "sourcing", {"product": "PX"})["best"] == {"region_b_share": 0.5}
    assert fake_clm.loads == 0


def test_the_news_query_gives_each_supplier_with_news(env):
    rows = run_query(env, "supplier_news", {"suppliers": ["SA1", "SA2", "SA3", "SB1", "SB2"]})["rows"]
    assert [row["name"] for row in rows] == ["Arden Plastics", "Bayline Components", "Borealis Motors"]
    assert "strike" in rows[2]["news"]


@pytest.fixture
def with_news(fake_clm):  # pylint: disable=unused-argument
    return run_scenario(load_environment(EXAMPLE), "sourcing_with_news", {"product": "PX"})


def test_the_strike_rejects_every_branch_buying_from_borealis(with_news):
    for share in (0.5, 1.0):
        rejected = [item for item in branch(with_news, share)["rejected_by"] if item["decision"] == "news_risk"]
        assert rejected and rejected[0]["inputs"] == {"probability": 0.9, "supplier": "Borealis Motors"}
    assert with_news["best"] == {"region_b_share": 0} and with_news["ranking"] == [{"region_b_share": 0}]


def test_each_supplier_in_use_with_news_is_judged(with_news):
    judged = [step["inputs"]["text"] for step in branch(with_news, 0)["steps"] if step["kind"] == "judgement"]
    assert len(judged) == 1 and judged[0].startswith("Arden Plastics")
    decided = [step for step in branch(with_news, 0)["steps"] if step["name"] == "news_risk"]
    assert decided[0]["inputs"] == {"probability": 0.1, "supplier": "Arden Plastics"}
    assert decided[0]["result"]["allowed"] is True


@pytest.mark.slow
def test_real_clm_rejects_the_strike():
    """With MLX on a Mac: CLM itself judges the graph's news, at the threshold judge_news.py measured."""
    pytest.importorskip("mlx")
    result = run_scenario(load_environment(EXAMPLE), "sourcing_with_news", {"product": "PX"})
    for share in (0.5, 1.0):
        assert "news_risk" in [item["decision"] for item in branch(result, share)["rejected_by"]]
    assert result["best"] == {"region_b_share": 0}
    print({share: [(step["inputs"]["text"][:30], step["result"]) for step in branch(result, share)["steps"]
                   if step["kind"] == "judgement"] for share in (0, 0.5, 1.0)})


def test_without_news_the_limit_is_decided_with_zero(fake_clm, monkeypatch):  # pylint: disable=unused-argument
    fresh = load_environment(EXAMPLE)
    monkeypatch.setattr(fresh.functions, "supplier_news", lambda graph, suppliers: [])
    result = run_scenario(fresh, "sourcing_with_news", {"product": "PX"})
    for item in result["branches"]:
        decided = [step for step in item["steps"] if step["name"] == "news_risk"]
        assert decided[0]["inputs"] == {"probability": 0.0, "supplier": "no supplier in use has news"}
    assert result["best"] == {"region_b_share": 0.5} and fake_clm.loads == 0
