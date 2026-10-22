"""Tests for evaluate.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import pytest
from conftest import FakeRouter, decision

import clm_router
from evaluate import agreement, confidence_bands, evaluate, format_report, main, match_report, suggest_threshold
from record import save_correction
from scenario import load_scenario


def route_record(state_text, clm, human):
    return {"kind": "route", "state_text": state_text, "human": human,
            "clm": {"route": clm, "confidence": 0.8, "probabilities": {clm: 0.9}}}


def test_suggest_threshold_is_the_lowest_that_reaches_the_target():
    pairs = [(0.2, False), (0.5, True), (0.7, True), (0.9, True)]
    assert suggest_threshold(pairs, target=0.9) == 0.5
    assert suggest_threshold([(0.9, False)], target=0.9) is None
    assert suggest_threshold([], target=0.9) is None


def test_confidence_bands_count_cases_and_put_one_in_the_top_band():
    bands = confidence_bands([(0.05, False), (0.65, True), (0.7, False), (1.0, True)])
    assert [(b["n"], b["correct"]) for b in bands] == [(1, 0), (0, 0), (0, 0), (2, 1), (1, 1)]
    assert (bands[3]["low"], bands[3]["high"]) == (0.6, 0.8)


def test_evaluate_replays_the_record_against_the_scenario(scenario):
    records = [
        route_record("User: refund", "Billing", "Billing"),
        route_record("User: crash", "Billing", "Tech"),
        route_record("User: slow", "Tech", "Tech"),
        route_record("User: parcel", "Tech", "Shipping"),            # route no longer in the scenario
    ]
    router = FakeRouter(
        {"User: refund": decision("Billing", 0.9), "User: crash": decision("Billing", 0.3),
         "User: slow": decision("Tech", 0.7)},
        tool_p=0.9,
        vectors={"User: refund": [1.0, 0.0], "User: crash": [0.0, 1.0], "User: slow": [0.0, 1.0]},
    )
    result = evaluate(records, scenario, router)

    routing = result["routing"]
    assert (routing["n"], routing["correct"], routing["skipped"]) == (3, 2, 1)
    assert routing["per_route"] == {"Billing": {"n": 1, "correct": 1}, "Tech": {"n": 2, "correct": 1}}
    assert routing["confusion"] == {("Billing", "Billing"): 1, ("Tech", "Billing"): 1, ("Tech", "Tech"): 1}
    assert routing["misses"] == [{"state_text": "User: crash", "human": "Tech", "clm": "Billing", "confidence": 0.3}]
    assert routing["suggested_threshold"] == 0.7
    assert set(result) == {"routing", "answers", "options", "matching"}
    # "User: slow" and "User: crash" are identical in vector space and the reviewer gave both the same route
    assert result["matching"] == {"n": 3, "matched": 2, "same_route": 2}


def test_match_report_excludes_a_case_matching_itself(scenario):
    records = [route_record("User: crash", "Billing", "Tech")]
    router = FakeRouter(decision("Billing"), vectors={"User: crash": [1.0, 0.0]})
    assert match_report(records, scenario, router) == {"n": 1, "matched": 0, "same_route": 0}


def test_report_names_the_numbers_a_reader_needs(scenario):
    records = [route_record("User: refund", "Billing", "Billing"), route_record("User: crash", "Billing", "Tech")]
    router = FakeRouter({"User: refund": decision("Billing", 0.9), "User: crash": decision("Billing", 0.3)},
                        vectors={"User: refund": [1.0, 0.0], "User: crash": [0.0, 1.0]})
    report = format_report(evaluate(records, scenario, router), scenario)

    assert "Routing: 1 of 2 correct (50%)" in report
    assert "Tech -> Billing  x1" in report
    assert "[0.30] Tech (CLM: Billing): User: crash" in report
    assert "Suggested confidence_threshold: 0.90 (current 0.60)" in report
    assert "Tool checks" not in report


def test_main_with_an_empty_record_explains_what_to_do(write_scenario, capsys):
    assert main([str(write_scenario())]) == 0
    assert "--review" in capsys.readouterr().out


def test_main_with_a_missing_scenario_exits_2(tmp_path, capsys):
    assert main([str(tmp_path / "nope.toml")]) == 2
    assert "scenario file not found" in capsys.readouterr().err


def test_scenario_path_is_required(capsys):
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 2
    assert "scenario" in capsys.readouterr().err


def test_evaluate_works_without_vectors_and_says_reuse_is_by_exact_text(scenario):
    class NoVectors(FakeRouter):
        exact_match = True

        def embed_state(self, state_text, scenario):
            raise AssertionError("a server backend has no vectors")

    records = [route_record("User: a", "Billing", "Billing"), route_record("User: b", "Billing", "Tech")]
    result = evaluate(records, scenario, NoVectors(decision("Billing")))
    assert result["matching"] is None
    assert "exact text" in format_report(result, scenario)


def test_agreement_counts(scenario):
    records = [route_record("refund", "Billing", "Billing"), route_record("crash", "Billing", "Tech"),
               route_record("parcel", "Tech", "Shipping")]          # route no longer in the scenario
    router = FakeRouter({"refund": decision("Billing"), "crash": decision("Billing")})
    assert agreement(records, scenario, router) == {"n": 2, "agree": 1}


def test_main_when_clm_cannot_load_is_one_line(write_scenario, monkeypatch, capsys):
    path = write_scenario()
    save_correction(load_scenario(path), "charged twice", "Billing", {"route": "Tech", "confidence": 0.7})

    def no_cuda(*args):
        raise clm_router.ClmError("CLM was set to run on cuda, which this machine does not have (available: cpu)")
    monkeypatch.setattr(clm_router, "load_engine", no_cuda)
    assert main([str(path)]) == 2
    assert capsys.readouterr().err.strip() == \
        "error: CLM was set to run on cuda, which this machine does not have (available: cpu)"


def test_evaluate_reports_corrected_answers_per_class(scenario):
    def urgent(value):
        return {"urgent": {"type": "noul", "noul": value}}
    records = [dict(route_record("User: refund", "Billing", "Billing"), answers={"urgent": True}),
               dict(route_record("User: crash", "Tech", "Tech"), answers={"urgent": True}),
               dict(route_record("User: slow", "Tech", "Tech"), answers={"urgent": False})]
    router = FakeRouter({r["state_text"]: decision(r["human"]) for r in records},
                        asks=[urgent(0.9), urgent(0.2), urgent(0.1)],
                        vectors={r["state_text"]: [1.0, 0.0] for r in records})
    result = evaluate(records, scenario, router)
    assert result["answers"] == {"urgent": {"true": [1, 2], "false": [1, 1]}}
    assert "urgent" in format_report(result, scenario) and "1 of 2" in format_report(result, scenario)
