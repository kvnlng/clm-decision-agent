"""Check that every figure in an agent's answer is one the tools returned, allowing for rounding and units.

A figure in the answer matches a tool's number when the tool's number, in the answer's unit and rounded to the
answer's decimals, is the figure: "$1.2 million" matches 1234567, "18.4%" matches 0.1843 or 18.43. Halves round
up, and signs are not compared ("fell by 12%" matches -0.12). A lowercase "m" is millions or metres, in the answer
or in a tool's text, so "$3m" matches 3000000 and "400 m" matches 400. Not checked: digits inside a word (Q3,
A100, 12.5kg), dates and times, years from 1900 to 2100 written without a comma, and bare whole numbers from 0 to 10
("3 branches").
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# The digits are atomic, so "12.5kg" is digits inside a word, never "12" before an unknown unit
_FIGURE = re.compile(r"(?<![\w.])(\$)?((?>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?))"
                     r"(?:\s*(%|k|m|M|million|bn|billion)\b|\s*(%))?(?![\w.,]\d|\w)")
# "m" is millions ("$3m") or metres ("400 m"), so either matches
_SCALES = {"k": (1e3,), "m": (1e6, 1.0), "M": (1e6,), "million": (1e6,), "bn": (1e9,), "billion": (1e9,),
           "%": (1.0, 0.01)}
# Dates and times are not figures: 2026-10-29, 16:07
_DATES = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}")


@dataclass(frozen=True)
class Number:
    """A figure as written: its text, its value as shown, the scales it may be in, and its decimals."""
    text: str
    value: float
    scales: tuple[float, ...]
    decimals: int


def figures(text: str) -> list[Number]:
    """Every figure in the text, the ones the check does not look at too."""
    found = []
    for match in _FIGURE.finditer(_DATES.sub(" ", text)):
        digits = match.group(2).replace(",", "")
        unit = match.group(3) or match.group(4)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        found.append(Number(match.group(0).strip(), float(digits), _SCALES.get(unit, (1.0,)), decimals))
    return found


def _checked(number: Number) -> bool:
    plain = number.scales == (1.0,) and number.decimals == 0 and not number.text.startswith("$")
    # "2,000" is written as a figure, not a year
    return not (plain and "," not in number.text and (number.value <= 10 or 1900 <= number.value <= 2100))


def numbers_in(text: str) -> list[Number]:
    """The figures in the text that the check looks at."""
    return [number for number in figures(text) if _checked(number)]


def _values(result) -> list[float]:
    """Every number in a tool result: in dicts, lists and strings, at any depth."""
    if isinstance(result, dict):
        return [value for item in result.values() for value in _values(item)]
    if isinstance(result, (list, tuple)):
        return [value for item in result for value in _values(item)]
    if isinstance(result, bool):
        return []
    if isinstance(result, (int, float)):
        return [float(result)]
    if isinstance(result, str):
        return [number.value * scale for number in figures(result) for scale in number.scales]
    return []


def _rounded(value: float, decimals: int) -> float:
    """Rounded as people round: halves up, away from zero; nan for a value too long to round, or not finite, which
    then matches nothing."""
    try:
        return float(Decimal(repr(value)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return math.nan


def _matches(number: Number, value: float) -> bool:
    # Signs are compared loosely: "fell by 12%" is written without the minus of -0.12
    return any(abs(_rounded(abs(value) / scale, number.decimals) - number.value) < 1e-9 for scale in number.scales)


def unsupported(answer: str, results: list) -> list[str]:
    """The figures in the answer that match no number in the results, as written."""
    values = _values(results)
    return [number.text for number in numbers_in(answer) if not any(_matches(number, value) for value in values)]
