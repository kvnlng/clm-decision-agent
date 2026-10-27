"""Tests for new_scenario.py: the starter folder a new scenario begins from."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json

import pytest
from conftest import FakeRouter

import clm_router
import decision
import evaluate
from new_scenario import main
from scenario import load_scenario
from tools import load_tools


@pytest.fixture
def made(tmp_path):
    folder = tmp_path / "my-shop"
    assert main([str(folder)]) == 0
    return folder


def refund(urgent=0.9):
    return {"route": "Refund", "confidence": 0.9, "probabilities": {"Refund": 0.9, "Where_Is_It": 0.05, "Other": 0.05},
            "answers": {"urgent": {"type": "noul", "noul": urgent}}, "source": "clm"}


def test_the_folder_has_a_scenario_a_tools_module_and_cases(made):
    assert sorted(path.name for path in made.iterdir()) == ["my_shop_tools.py", "scenario.cases.jsonl",
                                                           "scenario.toml"]
    scenario = load_scenario(made / "scenario.toml")
    assert scenario.tools_module == "my_shop_tools"
    assert {tool.__name__ for tool in load_tools(scenario).values()} == {"order_status", "open_case"}


def test_the_mock_tools_inform_and_act(made):
    scenario = load_scenario(made / "scenario.toml")
    router = FakeRouter(refund(), asks=[{"urgent": {"type": "noul", "noul": 0.9}}])
    reply = decision.decide_many([{"text": "It arrived broken, I want my money back", "context": {"order": "A100"}}],
                                 scenario, router, load_tools(scenario))[0]
    assert any("A100" in fact for fact in reply["informed"][0]["facts"])
    assert "delivered 2 days ago" in router.states[-1]          # the route was decided with the order's facts
    assert reply["actions"][0]["tool"] == "open_case" and reply["actions"][0]["status"] == "done"


def test_the_live_backend_says_it_is_not_written_yet(made):
    text = (made / "scenario.toml").read_text(encoding="utf-8").replace('backend = "mock"', 'backend = "live"')
    (made / "scenario.toml").write_text(text, encoding="utf-8")
    scenario = load_scenario(made / "scenario.toml")
    router = FakeRouter(refund(), asks=[{"urgent": {"type": "noul", "noul": 0.9}}])
    reply = decision.decide_many([{"text": "Refund please", "context": {"order": "A100"}}], scenario, router,
                                 load_tools(scenario))[0]
    assert reply["actions"][0]["status"] == "failed" and "live" in reply["actions"][0]["error"]


def test_the_cases_are_scored_by_evaluate(made, monkeypatch, capsys):
    cases = [json.loads(line) for line in (made / "scenario.cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cases) >= 6 and {case["route"] for case in cases} == {"Refund", "Where_Is_It", "Other"}
    router = FakeRouter(refund(), asks=[{"urgent": {"type": "noul", "noul": 0.9}}] * len(cases))
    monkeypatch.setattr(clm_router, "make_router", lambda scenario: (router, "test heads"))
    assert evaluate.main([str(made / "scenario.toml"), "--cases", str(made / "scenario.cases.jsonl")]) == 0
    assert "Route: " in capsys.readouterr().out


def test_an_existing_folder_with_files_is_left_alone(tmp_path, capsys):
    folder = tmp_path / "shop"
    folder.mkdir()
    (folder / "scenario.toml").write_text("mine", encoding="utf-8")
    assert main([str(folder)]) == 2
    assert (folder / "scenario.toml").read_text(encoding="utf-8") == "mine"
    assert "not empty" in capsys.readouterr().err


def test_the_backend_suits_the_machine(tmp_path, monkeypatch):
    monkeypatch.setattr("new_scenario.mlx_available", lambda: False)
    assert main([str(tmp_path / "linux")]) == 0
    assert load_scenario(tmp_path / "linux" / "scenario.toml").clm_backend == "torch"
    monkeypatch.setattr("new_scenario.mlx_available", lambda: True)
    assert main([str(tmp_path / "mac")]) == 0
    assert load_scenario(tmp_path / "mac" / "scenario.toml").clm_backend == "mlx"
