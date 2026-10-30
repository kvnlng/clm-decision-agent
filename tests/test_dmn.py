"""Tests for dmn.py: DMN decision tables, the subset the framework uses."""
# Test names say what they check.
# pylint: disable=missing-function-docstring
from pathlib import Path

import pytest
from conftest import limit_dmn

from clm_decision.dmn import DmnError, evaluate, load_table

TABLES = Path(__file__).parent / "dmn"


def test_first_hit():
    table = load_table(TABLES / "quality.dmn")
    assert table.name == "quality_threshold" and table.inputs == ["defect_rate"] and table.types == ["number"]
    assert table.outputs == ["allowed", "reason"] and table.hit_policy == "FIRST"
    assert evaluate(table, {"defect_rate": 0.03}) == {
        "outputs": {"allowed": False, "reason": "The defect rate is above 2%"}, "fired": ["r1"]}
    assert evaluate(table, {"defect_rate": 0.01}) == {"outputs": {"allowed": True, "reason": "Within the limit"},
                                                     "fired": ["r2"]}


@pytest.mark.parametrize("inputs, fired", [
    ({"size": 4.9, "grade": "B", "count": 3}, "small"),
    ({"size": 5, "grade": "A", "count": 4}, "large"),
    ({"size": 1, "grade": "C", "count": 3}, "graded"),
    ({"size": 100, "grade": "A", "count": 3}, "three"),
])
def test_ranges_lists_and_not(inputs, fired):
    assert evaluate(load_table(TABLES / "forms.dmn"), inputs)["fired"] == [fired]


def test_collect_returns_every_match():
    result = evaluate(load_table(TABLES / "collect.dmn"), {"days": 6})
    assert result == {"outputs": [{"flag": "late"}, {"flag": "very late"}], "fired": ["late", "very_late"]}
    assert evaluate(load_table(TABLES / "collect.dmn"), {"days": 0}) == {"outputs": [], "fired": []}


def test_unique_with_two_matches_is_an_error():
    with pytest.raises(DmnError, match=r"quality_threshold.*r1.*r2"):
        evaluate(load_table(TABLES / "overlap.dmn"), {"defect_rate": 0.03})


def test_no_matching_rule_is_an_error():
    with pytest.raises(DmnError, match=r"quality_threshold: no rule matches .*defect_rate.*0\.01"):
        evaluate(load_table(TABLES / "gap.dmn"), {"defect_rate": 0.01})


def test_a_missing_input_is_named():
    with pytest.raises(DmnError, match="defect_rate"):
        evaluate(load_table(TABLES / "quality.dmn"), {})


def test_an_unsupported_expression_is_refused_at_load():
    with pytest.raises(DmnError, match=r'unsupported.dmn rule r1: unsupported expression "date\("2026-01-01"\)"'):
        load_table(TABLES / "unsupported.dmn")


@pytest.mark.parametrize("table, problem", [
    (limit_dmn("not()"), r'rule fine: unsupported expression "not\(\)"'),
    (limit_dmn("not(-)"), r'rule fine: unsupported expression "not\(-\)"'),
    (limit_dmn("[5..1]"), r'rule fine: unsupported expression "\[5\.\.1\]"'),
    (limit_dmn().replace("<text>true</text>", "<text></text>"), r"rule fine: empty entry"),
], ids=["not", "not_dash", "backwards_range", "empty_output"])
def test_a_bad_entry_is_refused_at_load(tmp_path, table, problem):
    (tmp_path / "bad.dmn").write_text(table, encoding="utf-8")
    with pytest.raises(DmnError, match="bad.dmn " + problem):
        load_table(tmp_path / "bad.dmn")


def test_a_number_test_on_text_is_an_error():
    with pytest.raises(DmnError, match="quality_threshold: defect_rate must be a number, not '0.9'"):
        evaluate(load_table(TABLES / "quality.dmn"), {"defect_rate": "0.9"})
