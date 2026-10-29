"""Tests for number_check.py: every figure in an answer must be one a tool returned."""
# Test names say what they check.
# pylint: disable=missing-function-docstring
from number_check import numbers_in, unsupported


def test_rounding_and_scale():
    assert unsupported("Margin rises to $1.2 million.", [{"q3_margin": 1234567}]) == []
    assert unsupported("Margin rises to $1.3 million.", [{"q3_margin": 1234567}]) == ["$1.3 million"]
    assert unsupported("Tariffs fall to $230k", [{"tariff": 230100.0}]) == []
    assert unsupported("It costs 1,795,100.", [1795100.0]) == []


def test_percentages():
    assert unsupported("a margin of 18.4%", [{"rate": 0.1843}]) == []
    assert unsupported("a margin of 18.4%", [{"rate": 18.43}]) == []
    assert unsupported("a margin of 19%", [{"rate": 0.1843}]) == ["19%"]


def test_ignored_numbers():
    assert numbers_in("Q3 2026, 3 branches, Product Line X, rule A100") == []
    assert [number.text for number in numbers_in("2 days and 14.5 days")] == ["14.5"]


def test_nested_results():
    results = [{"branches": [{"figures": {"q3_margin": 412000}}]}, "no numbers here", {"text": "lead 12.5"}]
    assert unsupported("The best margin is 412,000 with a 12.5-day lead time.", results) == []
    assert unsupported("The best margin is 413,000.", results) == ["413,000"]


def test_signs_halves_units_and_dates():
    assert unsupported("Margin fell by 12%, to -3.5", [{"change": -0.12, "margin": -3.5}]) == []
    assert unsupported("$1.3 million and 13 units", [1250000, 12.5]) == []
    assert unsupported("$1.2m", [1234567]) == []
    assert numbers_in("Run on 2026-10-29 at 16:07") == []
