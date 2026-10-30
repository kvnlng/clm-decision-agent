"""Tests for new_environment.py: clm-new-environment writes a small environment that loads and decides."""
# Test names say what they check.
# pylint: disable=missing-function-docstring
from clm_decision import engine
from clm_decision.environment import load_environment
from clm_decision.new_environment import main


def test_the_new_environment_loads_and_decides(tmp_path, capsys):
    folder = tmp_path / "team-offsite"
    assert main([str(folder)]) == 0
    assert "clm-call" in capsys.readouterr().out
    env = load_environment(folder / "environment.toml")
    result = engine.run_scenario(env, "offsite", {"team": "T1", "days": 2})
    branches = {branch["levers"]["venue"]: branch for branch in result["branches"]}
    assert result["best"] == {"venue": "V3"} and branches["V3"]["figures"]["total_cost"] == 6960
    assert branches["V2"]["status"] == "rejected"
    assert branches["V2"]["rejected_by"][0]["reason"] == "The venue is too small for everyone"
    assert (folder / "team_offsite_functions.py").exists()


def test_an_unknown_team_fails_rather_than_counting_nobody(tmp_path):
    main([str(tmp_path / "offsite")])
    env = load_environment(tmp_path / "offsite" / "environment.toml")
    result = engine.run_scenario(env, "offsite", {"team": "T9", "days": 2})
    assert {branch["status"] for branch in result["branches"]} == {"failed"} and result["best"] is None
    assert "no team T9" in result["branches"][0]["error"]


def test_a_folder_that_is_not_empty_is_refused(tmp_path, capsys):
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "notes.txt").write_text("mine")
    assert main([str(tmp_path / "taken")]) == 2
    assert "is not empty" in capsys.readouterr().err
