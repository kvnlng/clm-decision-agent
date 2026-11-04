"""The mechanical check of a derived rule or fact: what moves an entry from proposed to checked.

An entry passes only when its cited document is the edition in force, its quote is in the passages it cites, every
figure it states is in that quote, and it uses only the names its slot allows. Figures are read strictly, unlike
number_check.py: small numbers, ordinals and the words one to twenty count too, and only zero is exempt, since an
ineligible row's 0 means no amount rather than a figure. The check reads; it decides nothing, and a person still
approves or rejects (derived.set_status).
"""
from __future__ import annotations

import itertools
import re
from datetime import date
from pathlib import Path

from . import derived
from .dmn import DmnError, Table, evaluate, load_table
from .environment import _CHECK_TYPE, _DMN_TYPES, Environment, Slot, edition, in_force

_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
          "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty")
# Not inside a word or after a decimal point; after ".." (a range's upper bound) it counts
_FIGURE = re.compile(r"(?<!\w)(?<!\d\.)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\b(?:" + "|".join(_WORDS) + r")\b",
                     re.IGNORECASE)
# Not policy figures, so not checked in a rule's statement or reason: ISO dates, section references (§ 4, § 4(2),
# § 4(2) and (4), Section 6(1), or bare: 6(3), 5(2)(a), 401(k)), and a bare year from 1900 to 2100 that is not
# dollars, a percentage, a count of days, weeks, months or years, or a decimal
_EXEMPT = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|(?:§|\bSection)\s*\d+(?:\s*\(\d+\))?(?:\s*(?:and|or|,)\s*\(\d+\))*"
                     r"|\b\d+\([0-9a-z]\)(?:\([0-9a-z]\))?"
                     r"|(?<![$\d.,])\b(?:19\d\d|20\d\d|2100)\b(?!\.\d)(?!\s*%)(?!\s*(?:days?|weeks?|months?|years?)\b)",
                     re.IGNORECASE)
_QUOTES = " \t\n\"'“”‘’"


def numbers(text: str) -> list[float]:
    """Every figure in the text, in order: digits with commas or a decimal point (a percentage or an ordinal such
    as 90th reads as its number), and the words one to twenty."""
    found = []
    for match in _FIGURE.finditer(text or ""):
        word = match.group(0).lower()
        found.append(_WORDS.index(word) + 1 if word in _WORDS else float(word.replace(",", "")))
    return [int(value) if float(value).is_integer() else value for value in found]


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _source(env: Environment, slot: Slot, source: dict, on: date) -> tuple[list[str], list[str]]:
    """The failures of the source's edition and quote, and the cited passages' texts."""
    ident = source.get("document")
    document = env.store.document(slot.collection, ident or "", None) if env.store and ident else {}
    if not document:
        return [f"source: {slot.collection} holds no document {ident}"], []
    failures = []
    if not in_force(document, on):
        failures.append(f"source: {ident} is not in force on {on}")
    if edition(document) != source.get("edition"):
        failures.append(f"source: the edition is {source.get('edition')}, but {ident} took effect on "
                        f"{edition(document)}")
    labels = list(source.get("passages") or [])
    texts = {passage["label"]: _collapse(passage["text"]) for passage in document["passages"]}
    failures += [f"source: {ident} has no passage {label}" for label in labels if label not in texts]
    if not labels:
        failures.append("source: no passage is cited")
    cited = [texts[label] for label in labels if label in texts]
    pieces = [_collapse(piece).strip(_QUOTES) for piece in re.split(r"\.\.\.|…", source.get("quote") or "")]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        failures.append("quote: there is no quote")
    failures += [f"quote: {piece!r} is not in {ident} {', '.join(labels)}" for piece in pieces
                 if not any(_within(piece, text) for text in cited)]
    return failures, cited


def _within(piece: str, text: str) -> bool:
    """Whether the piece is in the text whole: never ending or starting inside a word or a number, so "up to 1"
    is not found in "up to 10"."""
    pattern = (r"(?<!\w)" if piece[0].isalnum() else "") + re.escape(piece) + (r"(?!\w)" if piece[-1].isalnum()
                                                                                else "")
    return re.search(pattern, text) is not None


def _figures(found: list[tuple[str, object]], quote: str) -> list[str]:
    """A failure for each figure, other than zero, that the quote does not state."""
    quoted = numbers(quote)
    failures = []
    for where, value in found:
        if isinstance(value, str) and where in ("statement", "reason"):
            value = _EXEMPT.sub(" ", value)
        figures = numbers(value) if isinstance(value, str) else [value]
        for figure in figures:
            if figure != 0 and not any(abs(figure - other) < 1e-9 for other in quoted):
                failures.append(f"figure {figure:g} ({where}) is not in the quote")
    return list(dict.fromkeys(failures))


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fact(env: Environment, slot: Slot, entry: dict) -> tuple[list[str], list[tuple[str, object]]]:
    failures = []
    node, prop, value = entry.get("node"), entry.get("property"), entry.get("value")
    if env.graph is None or node not in {item["id"] for item in env.graph.nodes()}:
        failures.append(f"ontology: the graph has no node {node}")
    if node != slot.node or prop != slot.property:
        failures.append(f"ontology: the slot is {slot.node}.{slot.property}, not {node}.{prop}")
    if not _CHECK_TYPE[slot.type](value):
        failures.append(f"ontology: the value {value!r} is not of type {slot.type}")
    figures = [("value", value)] if _numeric(value) or isinstance(value, str) else []
    return failures, figures


# pylint: disable-next=too-many-locals
def _decision(env: Environment, slot: Slot, entry: dict) -> tuple[list[str], list[tuple[str, object]]]:
    try:
        table = load_table(Path(env.path).resolve().parent / "derived" / entry.get("dmn", ""))
    except DmnError as error:
        return [f"ontology: the DMN does not load: {error}"], []
    failures = []
    inputs = {name: _DMN_TYPES.get(kind.rsplit(":", 1)[-1], "any") for name, kind in zip(table.inputs, table.types)}
    if inputs != slot.inputs:
        failures.append(f"ontology: the table's inputs {inputs} are not the slot's {slot.inputs}")
    missing = [name for name in (*slot.outputs, "citation") if name not in table.outputs]
    if missing:
        failures.append(f"ontology: the table has no output {', '.join(missing)}")
    source = (entry.get("source") or {}).get("document")
    rule = entry.get("rule") or {}
    cites = list(rule.get("cites") or [])
    if cites != [source]:
        failures.append(f"ontology: the rule cites {cites}, not only its source {source}")
    figures: list[tuple[str, object]] = [("statement", rule.get("statement", "")), ("reason", rule.get("reason", ""))]
    for row in table.rules:
        figures += [(f"row {row.id} {name}", text) for name, text in zip(table.inputs, row.entries)]
        for name, value in zip(table.outputs, row.values):
            if name == "citation":
                named = [part.strip() for part in str(value).split(";") if part.strip()]
                failures += [f"ontology: row {row.id} cites {other}, not its source {source}"
                             for other in named if other != source]
            elif _numeric(value) or isinstance(value, str):
                figures.append((f"row {row.id} {name}", value))
    return failures, figures


def check(env: Environment, kind: str, name: str, entry: dict, *, on: date) -> list[str]:
    """The reasons the entry fails the check, one line each; empty when it passes. `on` is the date its cited
    document must be in force."""
    slot = env.slots.get(name)
    if slot is None or slot.kind != kind:
        return [f"ontology: the environment has no {kind} slot {name}"]
    failures, _ = _source(env, slot, entry.get("source") or {}, on)
    found, figures = (_fact if kind == "facts" else _decision)(env, slot, entry)
    failures += found
    failures += [f"ontology: {item} is not in the ontology, so the entry cannot be used"
                 for item in entry.get("missing") or []]
    failures += _figures(figures, (entry.get("source") or {}).get("quote") or "")
    return failures


def _stale(env: Environment, name: str, entry: dict, on: date) -> str | None:
    """Why the entry's cited document is no longer the edition it was derived from, or None."""
    slot, source = env.slots.get(name), entry.get("source") or {}
    if slot is None or not env.store or not source.get("document"):
        return None
    ident = source["document"]
    document = env.store.document(slot.collection, ident, None)
    if not document:
        return f"the cited document {ident} is no longer in {slot.collection}"
    if edition(document) != source.get("edition"):
        return (f"the cited document has a newer edition: {ident} took effect on {edition(document)}, and the entry "
                f"was derived from the {source.get('edition')} edition")
    if not in_force(document, on):
        return f"the cited document {ident} is not in force on {on}"
    return None


def recheck(folder: str | Path, env: Environment, *, on: date) -> dict:
    """Check every proposed or checked entry again, as after a correction, recording each result through
    derived.set_check, and send any entry but a rejected one whose cited document is no longer the edition it came
    from back to proposed (derived.set_stale). Otherwise approved and rejected entries are a person's decision and
    are left alone. -> counts."""
    counts = {"checked": 0, "proposed": 0, "stale": 0, "untouched": 0}
    layer = derived.read(folder)
    for kind in ("decisions", "facts"):
        for name, entry in layer[kind].items():
            why = _stale(env, name, entry, on) if entry.get("status") != "rejected" else None
            if why:
                derived.set_stale(folder, kind, name, why)
                counts["stale"] += 1
                continue
            if entry.get("status") not in ("proposed", "checked"):
                counts["untouched"] += 1
                continue
            failures = check(env, kind, name, entry, on=on)
            done = derived.set_check(folder, kind, name, {"passed": not failures, "failures": failures})
            counts[done["status"]] += 1
    return counts


CLOSE = 0.005           # two readings of a number agree within 0.5%, as the handbook's answer scoring does
SHOWN = 3               # the points a disagreement's failure line lists
KEPT = 20               # the points check.second_reading keeps


def _bounds(table: Table, name: str) -> set[float]:
    if name not in table.inputs:
        return set()
    column = table.inputs.index(name)
    return {float(found) for row in table.rules for found in re.findall(r"-?\d+(?:\.\d+)?", row.entries[column])}


def _strings(table: Table, name: str) -> set[str]:
    if name not in table.inputs:
        return set()
    column = table.inputs.index(name)
    return {found for row in table.rules for found in re.findall(r'"([^"]*)"', row.entries[column])}


def grid(first: Table, second: Table) -> list[dict]:
    """The inputs two tables are compared on, every combination of: for a number, each bound in either table, one
    either side, and 0, never below 0 (a count of days or years); both booleans; for a string, each value either
    table names and one neither does."""
    axes = []
    for index, name in enumerate(first.inputs):
        kind = first.types[index].rsplit(":", 1)[-1]
        if kind == "boolean":
            axes.append([True, False])
        elif kind == "string":
            axes.append(sorted(_strings(first, name) | _strings(second, name)) + ["(none of these)"])
        else:
            bounds = _bounds(first, name) | _bounds(second, name)
            values = {0.0} | bounds | {bound + 1 for bound in bounds} | {bound - 1 for bound in bounds}
            axes.append(sorted(int(value) if value.is_integer() else value for value in values if value >= 0))
    return [dict(zip(first.inputs, values)) for values in itertools.product(*axes)]


def _outputs(table: Table, inputs: dict, names: list[str]):
    try:
        found = evaluate(table, {key: value for key, value in inputs.items() if key in table.inputs})["outputs"]
    except DmnError:
        return "no answer"
    return {name: found.get(name) for name in names}


def table_differences(first: Table, second: Table, names: list[str]) -> list[dict]:
    """Where two tables give different values of the named outputs on the grid, first's inputs deciding the grid's
    axes; a point where neither table has a row agrees."""
    differences = []
    for inputs in grid(first, second):
        one, other = _outputs(first, inputs, names), _outputs(second, inputs, names)
        if one != other:
            differences.append({"inputs": inputs, "first": one, "second": other})
    return differences


def same_value(first, second) -> bool:
    """Whether two readings of a fact agree: numbers within 0.5%, anything else equal and of the same kind."""
    if _numeric(first) and _numeric(second):
        return abs(first - second) <= CLOSE * max(abs(second), 1)
    return type(first) is type(second) and first == second


def _shown(value) -> str:
    if isinstance(value, dict):
        return (_shown(next(iter(value.values()))) if len(value) == 1
                else ", ".join(f"{name} {_shown(item)}" for name, item in value.items()))
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{value:g}" if _numeric(value) else str(value)


def disagreement(model: str, differences: list[dict]) -> str:
    """The failure line for a second reading that disagrees: where, and what the first reading gives against the
    second, for the first few points."""
    points = []
    for item in differences[:SHOWN]:
        where = ", ".join(f"{name}={_shown(value)}" for name, value in (item.get("inputs") or {}).items())
        points.append((f"at {where}, " if where else "") + f"{_shown(item['first'])} against {_shown(item['second'])}")
    more = f"; and at {len(differences) - SHOWN} more" if len(differences) > SHOWN else ""
    return f"a second reading by {model} disagrees: " + "; ".join(points) + more


def conflicts(env: Environment, layer: dict) -> dict:
    """Each derived fact whose node already has a supplied value that differs: an open conflict, where the
    supplied value stands."""
    found = {}
    for name, entry in layer["facts"].items():
        node, prop = entry.get("node"), entry.get("property")
        if env.graph is None or node not in {item["id"] for item in env.graph.nodes()} \
                or prop in env.derived_facts.get(node, {}):
            continue
        supplied = env.graph.node(node).get(prop)
        if supplied is not None and supplied != entry.get("value"):
            found[name] = {"target": "facts", "name": name, "supplied": supplied, "derived": entry.get("value"),
                           "source": entry.get("source"), "status": "open", "decided_by": None,
                           "decided_at": None, "note": None}
    return found
