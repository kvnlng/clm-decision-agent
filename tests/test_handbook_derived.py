"""Tests for the handbook example's two measured variants: environment_derived.toml, whose rules and plan figures
are slots derived from the handbook, and environment_documents_only.toml, which has the handbook and no rules. The
layers here are written by hand to test loading; nothing is derived."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import importlib.util
import json
import shutil
import tomllib
from datetime import date
from pathlib import Path

import pytest
from fake_documents import FakeEmbedder, FakeStore

from clm_decision import derive_check, derived
from clm_decision.engine import run_scenario
from clm_decision.environment import load_environment

EXAMPLE = Path(__file__).parent.parent / "examples" / "handbook"
KEY = tomllib.loads((EXAMPLE / "answer_key.toml").read_text(encoding="utf-8"))
TODAY = "2026-12-01"
ASKED = {"q1": ("match", {}, {"contribution_percent": 3}),
         "q2": ("carryover", {}, {"days_taken": 0}),
         "q3": ("parental_leave", {}, {"weeks_taken": 15}),
         "q4": ("tuition", {"tuition": 3900, "job_related": True}, {"calendar_year": 2027})}
# The handbook's figures, read by hand from handbook.jsonl § 4 (2) to (4), as a person approving the facts would
FACTS = {"irs_limit": (24500, "(2)", "which for 2026 is $24,500"),
         "match_cap": (6300, "(4)", "The company match is capped at $6,300 a plan year"),
         "first_tier_percent": (3, "(3)", "Meridian matches 100% of the first 3% of pay you contribute"),
         "first_tier_rate": (100, "(3)", "Meridian matches 100% of the first 3% of pay you contribute"),
         "second_tier_percent": (3, "(3)", "and 50% of the next 3%"),
         "second_tier_rate": (50, "(3)", "and 50% of the next 3%")}
CITES = {"match_eligible": "hb-401k-2026", "vesting": "hb-401k-2026", "carryover_limit": "hb-leave",
         "parental_eligible": "hb-parental", "tuition_eligible": "hb-tuition"}


def store() -> FakeStore:
    found = FakeStore()
    found.replace("handbook", [json.loads(line) for line in (EXAMPLE / "handbook.jsonl").read_text().splitlines()],
                  [], {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "embeddinggemma",
                       "dimensions": 26})
    return found


@pytest.fixture
def folder(tmp_path):
    """The example without its derived layer or runs, so every slot starts unfilled."""
    copy = tmp_path / "handbook"
    shutil.copytree(EXAMPLE, copy, ignore=shutil.ignore_patterns("derived", "runs", "__pycache__"))
    return copy


def load(path, name="environment_derived.toml"):
    return load_environment(path / name, store=store(), embedder=FakeEmbedder(), today=date(2026, 12, 1))


def functions():
    spec = importlib.util.spec_from_file_location("handbook_functions", EXAMPLE / "handbook_functions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_slot_description_holds_a_figure(folder):
    env = load(folder)
    assert len(env.slots) == 11
    assert {name: derive_check.numbers(slot.description) for name, slot in env.slots.items()
            if derive_check.numbers(slot.description)} == {}


def test_tiered_match_with_the_keys_tiers_gives_the_keys_figures():
    tiers = {name: FACTS[name][0] for name in ("first_tier_percent", "first_tier_rate", "second_tier_percent",
                                              "second_tier_rate")}
    match = functions().tiered_match
    assert match(salary=92000, contribution_percent=4, match_cap=6300, **tiers)["match"] == \
        KEY["q1"]["figures"]["company_match"]["value"]
    assert match(salary=92000, contribution_percent=8, match_cap=6300, **tiers)["match"] == \
        KEY["q1"]["figures"]["most_match"]["value"]
    assert match(salary=500000, contribution_percent=8, match_cap=6300, **tiers)["match"] == 6300


def test_the_variant_loads_with_every_slot_unfilled(folder):
    env = load(folder)
    assert sorted(env.unfilled) == sorted(env.slots) and not env.admitted
    assert not env.decisions and "plan" not in env.queries and "company_match" not in env.calculations


def test_an_unfilled_fact_fails_the_match_branch_with_its_reason(folder):
    result = run_scenario(load(folder), "match", {"employee": "e_2001", "as_of": TODAY})
    assert {branch["status"] for branch in result["branches"]} == {"failed"}
    assert "k401_2026 has no irs_limit" in result["branches"][0]["error"]
    assert "no admitted derived fact" in result["branches"][0]["error"]


def hand_written_layer(folder: Path) -> None:
    """The hand-written tables and figures as approved derived entries: a fixture for loading, not a derivation."""
    layer = derived.empty("handbook")
    (folder / "derived" / "rules").mkdir(parents=True)
    for name, document in CITES.items():
        shutil.copy(folder / "rules" / f"{name}.dmn", folder / "derived" / "rules" / f"{name}.dmn")
        layer["decisions"][name] = {
            "description": name, "dmn": f"rules/{name}.dmn", "collection": "handbook",
            "rule": {"statement": name, "reason": "hand-written fixture", "owner": "test", "version": "2026-01-01",
                     "governs": ["benefits"], "cites": [document]},
            "source": {"document": document, "passages": ["(1)"], "quote": "", "edition": "2026-01-01"},
            "status": "approved", "check": {"passed": True, "failures": []}}
    for name, (value, passage, quote) in FACTS.items():
        layer["facts"][f"k401_2026.{name}"] = {
            "node": "k401_2026", "property": name, "value": value, "status": "approved",
            "source": {"document": "hb-401k-2026", "passages": [passage], "quote": quote, "edition": "2026-01-01"},
            "check": {"passed": True, "failures": []}}
    derived.write(folder, layer)


@pytest.mark.parametrize("question", ["q1", "q2", "q3", "q4"])
def test_a_hand_written_fixture_layer_answers_q1_to_q4(folder, question):
    hand_written_layer(folder)
    env = load(folder)
    assert not env.unfilled
    scenario, params, levers = ASKED[question]
    result = run_scenario(env, scenario, {"employee": KEY[question]["employee"], "as_of": TODAY, **params})
    found = next(branch for branch in result["branches"] if branch["levers"] == levers)
    assert found["status"] == "kept", found["error"]
    for item, figure in KEY[question]["figures"].items():
        assert found["figures"][item] == pytest.approx(figure["value"]), item
    assert all(rule["origin"] == "derived" for rule in result["rules"].values())


def measure():
    spec = importlib.util.spec_from_file_location("handbook_measure", EXAMPLE / "measure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_second_key_finds_a_checked_table_that_is_wrong(folder, tmp_path):
    hand_written_layer(folder)
    snapshot = tmp_path / "runs" / "derived-test"
    shutil.copytree(folder / "derived", snapshot)
    carryover = snapshot / "rules" / "carryover_limit.dmn"
    carryover.write_text(carryover.read_text(encoding="utf-8").replace("<text>7</text>", "<text>8</text>"),
                         encoding="utf-8")
    layer = json.loads((snapshot / "derived.json").read_text(encoding="utf-8"))
    layer["facts"]["k401_2026.match_cap"].update(value=6000, check={"passed": False, "failures": ["no 6000"]})
    (snapshot / "derived.json").write_text(json.dumps(layer), encoding="utf-8")
    found = measure().compare(tmp_path / "runs")["test"]
    assert found["summary"]["checked_but_wrong"] == 1 and found["summary"]["right"] == 9
    assert found["summary"]["failed"] == 1 and found["summary"]["slots"] == 11
    assert not found["slots"]["carryover_limit"]["right"] and found["slots"]["carryover_limit"]["check_passed"]
    assert {"inputs": {"years_of_service": 0}, "hand": {"carryover_limit": 7}, "derived": {"carryover_limit": 8}} \
        in found["slots"]["carryover_limit"]["differences"]
    assert found["slots"]["k401_2026.first_tier_rate"]["right"]


def test_the_second_keys_grid_holds_no_negative_count(folder):
    from clm_decision.dmn import load_table  # pylint: disable=import-outside-toplevel
    table = load_table(folder / "rules" / "carryover_limit.dmn")
    grid = derive_check.grid(table, table)
    assert {row["years_of_service"] for row in grid} == {0, 4, 5, 6}


def test_a_kept_layer_is_checked_again_and_only_its_passing_entries_approved(folder, tmp_path):
    hand_written_layer(folder)
    layer = derived.read(folder)
    layer["facts"]["k401_2026.match_cap"]["source"]["quote"] = "The company match is capped"
    derived.write(folder, layer)
    kept = tmp_path / "runs" / "derived-test"
    shutil.copytree(folder / "derived", kept)
    module = measure()
    assert module.recheck_layer("test", tmp_path / "runs", folder, load(folder)) == 0
    entries = json.loads((kept / "derived.json").read_text(encoding="utf-8"))["facts"]
    assert entries["k401_2026.match_cap"]["status"] == "proposed"
    assert entries["k401_2026.irs_limit"]["status"] == "approved"
    assert entries["k401_2026.irs_limit"]["decided_by"] == module.SCRIPTED


def test_a_derived_setup_needs_the_layer_of_its_model(folder):
    module = measure()
    assert module.wrong_layer("derived", folder) and module.wrong_layer("derived_local", folder)
    hand_written_layer(folder)
    layer = derived.read(folder)
    derived.write(folder, {**layer, "model": "claude-opus-5-5"})
    assert module.wrong_layer("derived", folder) is None and "gemma4" in module.wrong_layer("derived_local", folder)
    assert module.wrong_layer("rules", folder) is None


def test_the_committed_layer_loads_and_its_approvals_say_they_are_scripted():
    env = load(EXAMPLE)
    layer = derived.read(EXAMPLE)
    assert len(env.admitted) + len(env.unfilled) == 11
    for kind in ("decisions", "facts"):
        for name, entry in layer[kind].items():
            assert entry["status"] in ("approved", "proposed")
            assert (entry["status"] == "approved") == (name in env.admitted) == entry["check"]["passed"]
            assert entry["status"] != "approved" or entry["decided_by"] == measure().SCRIPTED


def test_documents_only_offers_only_the_employee_query_calculations_and_documents(folder):
    env = load(folder, "environment_documents_only.toml")
    assert list(env.queries) == ["employee"] and not env.decisions and not env.scenarios and not env.slots
    assert sorted(env.calculations) == ["carryover_days", "leave_pay", "reimbursement", "tiered_match"]
    assert list(env.collections) == ["handbook"] and env.document_tools and env.cited_text
    assert "irs_limit" not in env.graph.node("k401_2026") and "match_cap" not in env.graph.node("k401_2026")
