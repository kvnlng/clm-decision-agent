"""DMN decision tables: the subset the framework uses, evaluated with no model and no dependency.

A table is read from a DMN XML file (any DMN namespace version). Each input's expression text is the name of the
variable it reads; each output's name is the key of the result. Supported:

    hit policies    UNIQUE (the default), FIRST, COLLECT
    input entries   -   a literal (1.5, "Region B", true)   < <= > >= = != with a number
                    ranges [a..b] (a..b) [a..b) (a..b]   comma-separated lists of these   not(...)
    output entries  number, string and boolean literals

Anything else is refused when the table is loaded, naming the file and rule, so a table never half-works.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree

HIT_POLICIES = ("UNIQUE", "FIRST", "COLLECT")
_NUMBER = r"-?\d+(?:\.\d+)?"
_COMPARE = re.compile(rf"(<=|>=|!=|<|>|=)\s*({_NUMBER})")
_RANGE = re.compile(rf"([\[(])\s*({_NUMBER})\s*\.\.\s*({_NUMBER})\s*([\])])")
_STRING = re.compile(r'"([^"]*)"')


class DmnError(ValueError):
    """A table that cannot be read, or an evaluation that has no answer."""


@dataclass(frozen=True)
class Rule:
    """One row: a test per input and a value per output."""
    id: str
    tests: tuple[Callable[[Any], bool], ...]
    values: tuple[Any, ...]


@dataclass(frozen=True)
class Table:
    """A decision table, ready to evaluate."""
    name: str
    inputs: list[str]
    outputs: list[str]
    hit_policy: str
    rules: list[Rule]


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _literal(text: str):
    """A number, string or boolean literal; raises ValueError for anything else."""
    text = text.strip()
    if re.fullmatch(_NUMBER, text):
        return float(text) if "." in text else int(text)
    if text in ("true", "false"):
        return text == "true"
    match = _STRING.fullmatch(text)
    if match:
        return match.group(1)
    raise ValueError(text)


def _split(text: str) -> list[str]:
    """Split at the commas that are not inside quotes, brackets or parentheses."""
    parts, depth, quoted, start = [], 0, False, 0
    for index, char in enumerate(text):
        if char == '"':
            quoted = not quoted
        elif not quoted and char in "[(":
            depth += 1
        elif not quoted and char in "])":
            depth -= 1
        elif not quoted and depth == 0 and char == ",":
            parts.append(text[start:index])
            start = index + 1
    return [part.strip() for part in parts + [text[start:]]]


def _compare(op: str, bound: float) -> Callable[[Any], bool]:
    checks = {"<": lambda v: v < bound, "<=": lambda v: v <= bound, ">": lambda v: v > bound,
              ">=": lambda v: v >= bound, "=": lambda v: v == bound, "!=": lambda v: v != bound}
    check = checks[op]
    return lambda value: _is_number(value) and check(value)


def _one(text: str) -> Callable[[Any], bool]:
    """One test of a list; raises ValueError for an unsupported one."""
    match = _COMPARE.fullmatch(text)
    if match:
        return _compare(match.group(1), float(match.group(2)))
    match = _RANGE.fullmatch(text)
    if match:
        low, high = float(match.group(2)), float(match.group(3))
        low_open, high_open = match.group(1) == "(", match.group(4) == ")"
        return lambda v: (_is_number(v) and (v > low if low_open else v >= low)
                          and (v < high if high_open else v <= high))
    expected = _literal(text)
    if _is_number(expected):
        return lambda value: _is_number(value) and value == expected
    return lambda value: type(value) is type(expected) and value == expected


def _test(text: str) -> Callable[[Any], bool]:
    """An input entry as a function of the input's value; raises ValueError for an unsupported one."""
    text = text.strip()
    if text in ("", "-"):
        return lambda _: True
    match = re.fullmatch(r"not\s*\((.*)\)", text)
    if match:
        inner = _test(match.group(1))
        return lambda value: not inner(value)
    tests = [_one(part) for part in _split(text)]
    return lambda value: any(test(value) for test in tests)


def _local(element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element, name: str) -> list:
    return [child for child in element if _local(child) == name]


def _text(element) -> str:
    texts = _children(element, "text")
    return (texts[0].text or "") if texts else ""


def _rules(path: Path, table, inputs: int, outputs: int) -> list[Rule]:
    rules = []
    for index, row in enumerate(_children(table, "rule")):
        rule_id = row.get("id") or f"rule {index + 1}"
        entries, values = _children(row, "inputEntry"), _children(row, "outputEntry")
        if len(entries) != inputs or len(values) != outputs:
            raise DmnError(f"{path.name} rule {rule_id}: needs {inputs} input and {outputs} output entries")
        try:
            tests = tuple(_test(_text(entry)) for entry in entries)
            results = tuple(_literal(_text(value)) for value in values)
        except ValueError as error:
            raise DmnError(f"{path.name} rule {rule_id}: unsupported expression {error}") from error
        rules.append(Rule(rule_id, tests, results))
    return rules


def load_table(path: Path) -> Table:
    """The first decision table in the file; raises DmnError naming the file (and rule) of any problem."""
    path = Path(path)
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise DmnError(f"{path.name}: cannot read: {error}") from error
    decisions = [element for element in root.iter() if _local(element) == "decision"
                 and any(_local(child) == "decisionTable" for child in element)]
    if not decisions:
        raise DmnError(f"{path.name}: no decision table")
    decision = decisions[0]
    table = _children(decision, "decisionTable")[0]
    name = decision.get("name") or decision.get("id") or path.stem
    hit_policy = table.get("hitPolicy", "UNIQUE")
    if hit_policy not in HIT_POLICIES:
        raise DmnError(f"{path.name}: hit policy {hit_policy} is not supported ({', '.join(HIT_POLICIES)})")
    inputs = [_text(_children(item, "inputExpression")[0]).strip() if _children(item, "inputExpression") else ""
              for item in _children(table, "input")]
    outputs = [item.get("name") or "" for item in _children(table, "output")]
    if not inputs or not all(inputs):
        raise DmnError(f"{path.name}: every input needs an inputExpression naming its variable")
    if not outputs or not all(outputs):
        raise DmnError(f"{path.name}: every output needs a name")
    return Table(name, inputs, outputs, hit_policy, _rules(path, table, len(inputs), len(outputs)))


def evaluate(table: Table, inputs: dict) -> dict:
    """-> {"outputs": ..., "fired": [rule ids]}; for COLLECT, outputs is a list of dicts, one per rule fired.
    Raises DmnError for a missing input, for no matching rule (UNIQUE, FIRST) or for two (UNIQUE)."""
    missing = [name for name in table.inputs if name not in inputs]
    if missing:
        raise DmnError(f"{table.name}: missing input {', '.join(missing)}")
    values = [inputs[name] for name in table.inputs]
    fired = [rule for rule in table.rules if all(test(value) for test, value in zip(rule.tests, values))]
    if table.hit_policy == "COLLECT":
        return {"outputs": [dict(zip(table.outputs, rule.values)) for rule in fired],
                "fired": [rule.id for rule in fired]}
    if not fired:
        shown = ", ".join(f"{name}={inputs[name]!r}" for name in table.inputs)
        raise DmnError(f"{table.name}: no rule matches {shown}")
    if table.hit_policy == "UNIQUE" and len(fired) > 1:
        raise DmnError(f"{table.name}: rules {', '.join(rule.id for rule in fired)} all match, and the hit policy "
                       "is UNIQUE")
    return {"outputs": dict(zip(table.outputs, fired[0].values)), "fired": [fired[0].id]}
