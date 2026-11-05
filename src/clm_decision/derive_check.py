"""The mechanical check of a derived rule or fact: what moves an entry from proposed to checked.

An entry passes only when its cited document is the edition in force, its quote is in the passages it cites, every
figure it states is in that quote, and it uses only the names its slot allows. Figures are read strictly, unlike
number_check.py: small numbers, ordinals and the words one to twenty count too, and only zero is exempt, since an
ineligible row's 0 means no amount rather than a figure. The check reads; it decides nothing, and a person still
approves or rejects (derived.set_status).
"""
from __future__ import annotations

import itertools
import math
import re
from datetime import date
from pathlib import Path
from typing import Callable

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
    if isinstance(value, list):     # each item of a list states its own figures
        return failures, [(f"value[{index}]", item) for index, item in enumerate(value)
                          if _numeric(item) or isinstance(item, str)]
    return failures, [("value", value)] if _numeric(value) or isinstance(value, str) else []


# pylint: disable-next=too-many-locals
def _decision(env: Environment, slot: Slot, entry: dict) -> tuple[list[str], list[tuple[str, object]]]:
    try:
        table = load_table(Path(env.path).resolve().parent / "derived" / entry.get("dmn", ""))
    except DmnError as error:
        return [f"ontology: the DMN does not load: {error}"], []
    failures = []
    inputs = {name: _DMN_TYPES.get(kind.rsplit(":", 1)[-1], "any") for name, kind in zip(table.inputs, table.types)}
    # A table states only number; an integer input is one too
    if inputs != {name: _DMN_TYPES.get(kind, kind) for name, kind in slot.inputs.items()}:
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


SecondReading = Callable[[str, str, dict], tuple[dict, list[str]]]


def _earlier(record: dict | None) -> list[str]:
    """The failure an earlier second reading still makes when none is made now: a disagreement or an unusable
    reading stands until a later reading agrees or a person corrects the entry, which clears its check."""
    if not record or record.get("agreed"):
        return []
    model = record.get("model")
    if record.get("error"):
        return [f"an earlier second reading by {model} could not be made: {record['error']}"]
    return ["an earlier " + disagreement(model, record.get("differences") or []).removeprefix("a ")]


def _reading(record: dict | None) -> str:
    return "not_made" if not record or record.get("error") else "agreed" if record.get("agreed") else "disagreed"


def _read_again(env: Environment, kind: str, name: str, entry: dict,
                second: SecondReading | None) -> tuple[dict | None, list[str]]:
    """The second reading's record and failures for a check: made now with `second`, but not for an entry whose
    slot is gone, which the check already fails; without `second`, only an earlier reading that did not agree, which
    still stands. An agreement compared the entry as it was then, so it is never carried into a later check."""
    if second is None:
        earlier = (entry.get("check") or {}).get("second_reading")
        return (None, []) if not earlier or earlier.get("agreed") else (earlier, _earlier(earlier))
    slot = env.slots.get(name)
    if slot is None or slot.kind != kind:
        return None, []
    return second(kind, name, entry)


def _still_stale(entry: dict, why: str) -> bool:
    """Whether the entry was already sent back to proposed for this reason, and not checked since."""
    return entry.get("status") == "proposed" and (entry.get("check") or {}).get("failures") == [why]


def recheck(folder: str | Path, env: Environment, *, on: date, second: SecondReading | None = None) -> dict:
    """Check every proposed or checked entry again, as after a correction, recording each result through
    derived.set_check, and send any entry but a rejected one whose cited document is no longer the edition it came
    from back to proposed (derived.set_stale); an approved one whose table changed since its check goes back to
    proposed too, and is checked again at once, as is, once, an approved one checked before table digests were kept
    (counted as undigested), so its digest comes from a check. Otherwise approved and rejected entries are a
    person's decision and are left alone. With `second` (kind, name, entry -> record, failures), each entry checked
    is also read again by a second model, whether or not the mechanical check passed it, and its failures join the
    check's (_read_again). An entry still stale for the same reason is counted, not sent back again. -> counts."""
    counts = {"checked": 0, "proposed": 0, "stale": 0, "untouched": 0, "undigested": 0}
    if second is not None:
        counts.update(agreed=0, disagreed=0, not_made=0)
    layer = derived.read(folder)
    for kind in ("decisions", "facts"):
        for name, entry in layer[kind].items():
            why = _stale(env, name, entry, on) if entry.get("status") != "rejected" else None
            if why:     # once only: while it is still stale for this reason, it is left as it is
                if not _still_stale(entry, why):
                    derived.set_stale(folder, kind, name, why)
                counts["stale"] += 1
                continue
            if entry.get("status") == "approved" and derived.changed_since_check(folder, entry):
                entry = derived.set_stale(folder, kind, name, "its DMN has changed since its check")
                counts["stale"] += 1
            elif entry.get("status") == "approved" and entry.get("dmn") and \
                    not (entry.get("check") or {}).get("dmn_sha256"):
                entry = derived.set_stale(folder, kind, name, "its check has no digest of its DMN, as it was made "
                                                              "before digests were kept")
                counts["undigested"] += 1
            if entry.get("status") not in ("proposed", "checked"):
                counts["untouched"] += 1
                continue
            record, read = _read_again(env, kind, name, entry, second)
            failures = check(env, kind, name, entry, on=on) + read
            if second is not None:
                counts[_reading(record)] += 1
            done = derived.set_check(folder, kind, name, {"passed": not failures, "failures": failures,
                                                          "second_reading": record})
            counts[done["status"]] += 1
    return counts


CLOSE = 0.005           # two readings of a number agree within 0.5%, as the handbook's answer scoring does
SHOWN = 3               # the points a disagreement's failure line lists
KEPT = 20               # the points check.second_reading keeps
MAX_POINTS = 100_000    # the most points two tables are compared on; more and the second reading is not made


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


def _points(bounds: set[float]) -> list:
    """Each bound and a point just either side of it, a step of half the gap to its nearest neighbour but at most 1,
    so rows that differ only between bounds less than 1 apart are compared too; one more below the lowest bound and
    above the highest, and 0. Negative values are taken too."""
    ordered = sorted(bounds)
    values = {0.0}
    for index, bound in enumerate(ordered):
        gaps = [abs(bound - other) for other in ordered[max(index - 1, 0):index + 2] if other != bound]
        step = min([1.0, *(gap / 2 for gap in gaps)])
        values |= {bound - step, bound, bound + step}
    if ordered:
        values |= {ordered[0] - 1, ordered[-1] + 1}
    rounded = {round(value, 9) for value in values}
    return sorted(int(value) if value.is_integer() else value for value in rounded)


def _whole_points(bounds: set[float]) -> list[int]:
    """For an integer input: the whole numbers at and either side of each bound (a bound between two, both), one
    below the lowest and one above the highest, and 0; none below 0 unless a bound is."""
    values = {0}
    for bound in bounds:
        values |= {math.floor(bound) - 1, math.floor(bound), math.ceil(bound), math.ceil(bound) + 1}
    lowest = min(bounds, default=0)
    return sorted(value for value in values if value >= 0 or lowest < 0)


def grid(first: Table, second: Table, kinds: dict[str, str] | None = None) -> list[dict]:
    """The inputs two tables are compared on, every combination of: for a number, the points around each bound in
    either table (_points), or for an input `kinds` (the slot's inputs) names an integer, whole numbers only
    (_whole_points); both booleans; for a string, each value either table names and one neither does. Raises
    DmnError, before building it, when it would have more than MAX_POINTS points."""
    axes = []
    for index, name in enumerate(first.inputs):
        kind = first.types[index].rsplit(":", 1)[-1]
        if kind == "boolean":
            axes.append([True, False])
        elif kind == "string":
            axes.append(sorted(_strings(first, name) | _strings(second, name)) + ["(none of these)"])
        else:
            bounds = _bounds(first, name) | _bounds(second, name)
            axes.append(_whole_points(bounds) if (kinds or {}).get(name) == "integer" else _points(bounds))
    size = math.prod(len(axis) for axis in axes)
    if size > MAX_POINTS:
        raise DmnError(f"the tables' inputs give {size} points to compare, more than {MAX_POINTS}")
    return [dict(zip(first.inputs, values)) for values in itertools.product(*axes)]


def _outputs(table: Table, inputs: dict, names: list[str]):
    try:
        found = evaluate(table, {key: value for key, value in inputs.items() if key in table.inputs})["outputs"]
    except DmnError:
        return "no answer"
    return {name: found.get(name) for name in names}


def _compared(outputs):
    """Outputs as they are compared: text in any case, without the spaces around it."""
    if not isinstance(outputs, dict):
        return outputs
    return {name: value.strip().casefold() if isinstance(value, str) else value for name, value in outputs.items()}


def table_differences(first: Table, second: Table, names: list[str],
                      kinds: dict[str, str] | None = None) -> list[dict]:
    """Where two tables give different values of the named outputs on the grid, first's inputs deciding the grid's
    axes and `kinds` (the slot's inputs) which are integers, text compared in any case and without the spaces around
    it; a point where neither table has a row agrees."""
    differences = []
    for inputs in grid(first, second, kinds):
        one, other = _outputs(first, inputs, names), _outputs(second, inputs, names)
        if _compared(one) != _compared(other):
            differences.append({"inputs": inputs, "first": one, "second": other})
    return differences


def same_value(first, second) -> bool:
    """Whether two readings of a fact agree: numbers within 0.5% of the larger, either way round (0 agrees only with
    0), anything else equal and of the same kind."""
    if _numeric(first) and _numeric(second):
        return abs(first - second) <= CLOSE * max(abs(first), abs(second))
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
    supplied value stands. A reading that gave no value, or one a person rejected, derived nothing to differ."""
    found = {}
    for name, entry in layer["facts"].items():
        if "value" not in entry or entry.get("status") == "rejected":
            continue
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
