"""Check that every figure in an agent's answer is one the tools returned, allowing for rounding and units.

A figure in the answer matches a tool's number when the tool's number, in the answer's unit and rounded to the
answer's decimals, is the figure: "$1.2 million" matches 1234567, "18.4%" matches 0.1843 or 18.43. Not checked:
digits inside a word (Q3, A100), years from 1900 to 2100, and bare whole numbers from 0 to 10 ("3 branches").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_FIGURE = re.compile(r"(?<![\w.])(\$)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
                     r"(?:\s*(%|k|M|million|bn|billion)\b|\s*(%))?(?![\w])")
_SCALES = {"k": (1e3,), "M": (1e6,), "million": (1e6,), "bn": (1e9,), "billion": (1e9,), "%": (1.0, 0.01)}


@dataclass(frozen=True)
class Number:
    """A figure as written: its text, its value as shown, the scales it may be in, and its decimals."""
    text: str
    value: float
    scales: tuple[float, ...]
    decimals: int


def _figures(text: str) -> list[Number]:
    found = []
    for match in _FIGURE.finditer(text):
        digits = match.group(2).replace(",", "")
        unit = match.group(3) or match.group(4)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        found.append(Number(match.group(0).strip(), float(digits), _SCALES.get(unit, (1.0,)), decimals))
    return found


def _checked(number: Number) -> bool:
    plain = number.scales == (1.0,) and number.decimals == 0 and not number.text.startswith("$")
    return not (plain and (number.value <= 10 or 1900 <= number.value <= 2100))


def numbers_in(text: str) -> list[Number]:
    """The figures in the text that the check looks at."""
    return [number for number in _figures(text) if _checked(number)]


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
        return [number.value * number.scales[0] for number in _figures(result)]
    return []


def _matches(number: Number, value: float) -> bool:
    return any(abs(round(value / scale, number.decimals) - number.value) < 1e-9 for scale in number.scales)


def unsupported(answer: str, results: list) -> list[str]:
    """The figures in the answer that match no number in the results, as written."""
    values = _values(results)
    return [number.text for number in numbers_in(answer) if not any(_matches(number, value) for value in values)]
