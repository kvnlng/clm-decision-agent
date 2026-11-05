"""clm-derive: derive the rules and facts an environment needs from its documents, check them, and record a person's
decision on each.

    clm-derive run ENVIRONMENT COLLECTION [--backend claude|ollama] [--model NAME] [--as-of DATE] [--fresh]
    clm-derive check ENVIRONMENT [--as-of DATE]
    clm-derive status ENVIRONMENT KIND NAME STATUS --by WHO [--note TEXT]

run fills each slot of the environment's [derived] table (docs/environment.md) that is not already approved or
rejected on an unchanged edition: it searches the collection with the slot's description, gives the model the
documents in force on --as-of (default today) whole, renders the decision table the model describes, and checks the
entry (derive_check.py), asking the model once more with the failures if it fails. Entries are written proposed, or
checked when they pass, to derived/derived.json; a derived fact that differs from a supplied one is an open conflict,
and the supplied value stands. check re-runs the check on every proposed or checked entry, as after a correction.
When [derived] names a second_reader, run and check also make a second reading of each entry they check: that
model, which must not be the layer's, derives it again from its cited document without seeing the first answer,
and an entry it disagrees with, or could not read, fails the check (second_reading).
status is a person's decision: approved, rejected, proposed (to check it again), or a conflict resolved or open;
only the mechanical check (run or check) sets checked. The model is Claude (the agent extra) or a local model through
Ollama, as clm-agent reaches them; errors are one line and a non-zero exit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import tomllib
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Protocol
from xml.sax.saxutils import escape

import httpx

from . import derive_check, derived
from .decision_agent import LOCAL_MODEL, MODEL, OLLAMA_URL
from .dmn import DmnError, load_table
from .documents import document_settings
from .engine import Steps
from .environment import Environment, Slot, edition, in_force, load_environment, second_reader_of
from .scenario import ScenarioError, StepError

RUN = "clm-derive run"
MAX_TOKENS = 8192
MAX_DOCUMENTS = 5
_JSON = {"number": {"type": "number"}, "integer": {"type": "integer"}, "boolean": {"type": "boolean"},
         "string": {"type": "string"}, "list": {"type": "array", "items": {"type": "string"}}}
_DMN = {"number": "number", "integer": "number", "boolean": "boolean", "string": "string", "list": "string"}
SYSTEM = """You derive one rule or one fact from a company's documents, for a decision engine that will apply it \
exactly as you write it. Use only the documents given, and only an edition in force on the date given. Answer in \
JSON with the schema.

- source: the document's id (as its "Document id" line gives it, such as hb-leave) for the one document the rule \
or fact comes from, the labels of the passages you used, such as "(2)", and a quote: the exact words of those passages, copied character for character. Join pieces from different places with \
"...".
- Every number you state, in the statement, in each row's tests, outputs and reason, or as a fact's value, must \
appear in the quote, as a numeral or a word ("one full year" is 1). 0 is exempt, for "none".
- A row's input test is one of: - (any value); a comparison such as >= 5 or < 90; a range such as [1..3]; true or \
false. Rows are tried in order and the first that matches applies, so put the most specific first and end with a \
row whose tests are all -.
- If the rule needs an input or a property the slot does not offer, name it in missing rather than inventing one."""


class DeriveError(RuntimeError):
    """A model that cannot be reached or used; the message is one line saying what to do."""


class AnswerError(DeriveError):
    """A model's answer that is not JSON, as when it ran out of tokens: a failure of that answer, not of the run."""


class Extractor(Protocol):  # pylint: disable=too-few-public-methods
    """A model that answers a prompt with JSON in the given schema."""
    model: str

    def extract(self, system: str, prompt: str, schema: dict) -> dict:
        """The model's answer, as the schema describes."""


def _first_line(error: Exception) -> str:
    return (str(error).strip().splitlines() or [type(error).__name__])[0]


class ClaudeExtractor:  # pylint: disable=too-few-public-methods
    """Claude through the Anthropic SDK, with structured output."""

    def __init__(self, client, model: str = MODEL):
        self.client, self.model = client, model

    def extract(self, system: str, prompt: str, schema: dict) -> dict:
        """The answer; raises DeriveError in one line when Claude cannot give one, and AnswerError, as Ollama's
        extractor does, when its answer is not JSON."""
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=16000, system=system, messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": schema}})
        except Exception as error:  # pylint: disable=broad-exception-caught  # the SDK raises many kinds
            raise DeriveError(f"Claude ({self.model}): {_first_line(error)}") from error
        try:
            return json.loads(next(block.text for block in response.content if block.type == "text"))
        except (StopIteration, ValueError) as error:
            raise AnswerError(f"Claude ({self.model}) did not answer in JSON: {_first_line(error)}") from error


class OllamaExtractor:  # pylint: disable=too-few-public-methods
    """A local model through Ollama's /api/chat, its output constrained to the schema."""

    def __init__(self, model: str = LOCAL_MODEL, url: str = OLLAMA_URL, http=None):
        self.model, self.url, self.http = model, url.rstrip("/"), http

    def extract(self, system: str, prompt: str, schema: dict) -> dict:
        """The answer; raises DeriveError in one line when Ollama or the model is not there."""
        # Without thinking: gemma4 thinking under a schema ran past 100,000 tokens without answering; the cap stops
        # any runaway, and its cut-off answer fails as not JSON
        body = {"model": self.model, "stream": False, "format": schema, "think": False,
                "options": {"temperature": 0, "num_predict": MAX_TOKENS},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        try:
            if self.http is None:
                with httpx.Client(timeout=1800) as http:
                    response = http.post(f"{self.url}/api/chat", json=body)
            else:
                response = self.http.post(f"{self.url}/api/chat", json=body)
        except httpx.HTTPError as error:
            raise DeriveError(f"cannot reach Ollama at {self.url}: {_first_line(error)}; is `ollama serve` "
                              "running?") from error
        if response.status_code == 404:
            raise DeriveError(f"Ollama has no model {self.model}: run `ollama pull {self.model}`")
        if response.status_code != 200:
            raise DeriveError(f"Ollama at {self.url} answered {response.status_code}")
        try:
            return json.loads(response.json()["message"]["content"])
        except (KeyError, ValueError) as error:
            raise AnswerError(f"{self.model} did not answer in JSON: {_first_line(error)}") from error


def _object(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(properties),
            "properties": properties}


def answer_schema(slot: Slot) -> dict:
    """What the model returns for the slot: a fact's value, or a decision's statement, reason and rows; with the
    source and anything missing from the ontology."""
    source = _object({"document": {"type": "string"}, "passages": {"type": "array", "items": {"type": "string"}},
                      "quote": {"type": "string"}})
    missing = {"type": "array", "items": {"type": "string"}}
    if slot.kind == "facts":
        return _object({"value": _JSON[slot.type], "source": source, "missing": missing})
    row = _object({"inputs": _object({name: {"type": "string"} for name in slot.inputs}),
                   "outputs": _object({name: _JSON[kind] for name, kind in slot.outputs.items()}),
                   "reason": {"type": "string"}})
    return _object({"statement": {"type": "string"}, "reason": {"type": "string"},
                    "rows": {"type": "array", "items": row}, "source": source, "missing": missing})


def candidates(env: Environment, slot: Slot, on: date) -> list[dict]:
    """The documents the slot's description finds in its collection that are in force on the date, whole, nearest
    first: what the model is given, so it never sees a replaced edition."""
    rows = Steps(env).search(slot.collection, slot.description, None, 20)
    found = []
    for ident in dict.fromkeys(row["document"] for row in rows):
        document = env.store.document(slot.collection, ident, None)
        if document and in_force(document, on):
            found.append(document)
    return found[:MAX_DOCUMENTS]


def _prompt(slot: Slot, documents: list[dict], on: date) -> str:
    if slot.kind == "facts":
        wanted = [f"A fact: the {slot.property} of {slot.node}, of type {slot.type}."]
    else:
        wanted = ["A decision table.",
                  "Inputs: " + ", ".join(f"{name} ({kind})" for name, kind in slot.inputs.items()) + ".",
                  "Outputs: " + ", ".join(f"{name} ({kind})" for name, kind in slot.outputs.items())
                  + "; every row also gives a reason in words. The citation is added for you."]
    shown = []
    for document in documents:
        dates = ", ".join(f"{key} {value}" for key, value in (document.get("metadata") or {}).items()
                          if key.startswith("effective"))
        shown.append(f"Document id: {document['id']}\n{document['citation']}, {document['title']} ({dates})\n"
                     + "\n".join(f"{passage['label']} {passage['text']}" for passage in document["passages"]))
    return "\n".join([f"Slot: {slot.name}", f"What it is: {slot.description}", *wanted, f"In force on: {on}", "",
                      "Documents:", "", "\n\n".join(shown)])


def _literal(value) -> str:
    """A table cell for an output's value: a number whole, never with an exponent or rounded, and a list as its
    items joined with "; "."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else format(Decimal(repr(value)), "f")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        value = "; ".join(str(item) for item in value)
    return '"' + str(value).replace('"', "'") + '"'


def render_dmn(slot: Slot, rows: list[dict], document: str) -> str:
    """The decision table the model described, in the DMN the framework reads: the slot's inputs and outputs, a
    reason, and a citation naming the document, first match applying."""
    outputs = {**slot.outputs, "reason": "string", "citation": "string"}
    head = "".join(f'<input id="{slot.name}_in{index}" label="{name}"><inputExpression typeRef="{_DMN[kind]}">'
                   f"<text>{name}</text></inputExpression></input>\n      "
                   for index, (name, kind) in enumerate(slot.inputs.items(), start=1))
    head += "".join(f'<output id="{slot.name}_out{index}" name="{name}" typeRef="{_DMN[kind]}"/>\n      '
                    for index, (name, kind) in enumerate(outputs.items(), start=1))
    body = ""
    for index, row in enumerate(rows, start=1):
        values = {**row["outputs"], "reason": row["reason"], "citation": document}
        body += (f'<rule id="{slot.name}_r{index}">'
                 + "".join(f"<inputEntry><text>{escape(str(row['inputs'][name]).strip() or '-')}</text></inputEntry>"
                           for name in slot.inputs)
                 + "".join(f"<outputEntry><text>{escape(_literal(values[name]))}</text></outputEntry>"
                           for name in outputs) + "</rule>\n      ")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!-- Derived by clm-derive from ' + escape(document)
            + '; see derived.json for its source and status. -->\n'
            f'<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" id="{slot.name}_definitions" '
            f'name="{slot.name}" namespace="https://example.org/derived">\n'
            f'  <decision id="{slot.name}" name="{slot.name}">\n'
            f'    <decisionTable id="{slot.name}_table" hitPolicy="FIRST">\n      {head}{body.rstrip()}\n'
            "    </decisionTable>\n  </decision>\n</definitions>\n")


def _blank(slot: Slot) -> dict:
    entry = {"description": slot.description} if slot.kind == "decisions" else {"node": slot.node,
                                                                                 "property": slot.property}
    return {**entry, "status": "proposed", "check": None, "decided_by": None, "decided_at": None, "note": None}


RULES, ASIDE = "rules", "rules.partial"      # in derived/: the tables, and where a run writes them until it ends


def _entry(env: Environment, slot: Slot, answer: dict) -> dict:
    """The answer as a layer entry, its decision table written aside, to derived/rules.partial/, until the run ends
    (_replace); raises KeyError or TypeError for an answer not in the schema."""
    given = answer["source"]
    document = env.store.document(slot.collection, str(given["document"]), None) or {}
    source = {"document": given["document"], "passages": list(given["passages"]), "quote": given["quote"],
              "edition": edition(document) if document else None}
    entry = {**_blank(slot), "source": source, "missing": list(answer.get("missing") or []),
             "collection": slot.collection}
    if slot.kind == "facts":
        return {**entry, "value": answer["value"]}
    aside = env.path.resolve().parent / "derived" / ASIDE
    aside.mkdir(parents=True, exist_ok=True)
    (aside / f"{slot.name}.dmn").write_text(render_dmn(slot, answer["rows"], str(given["document"])),
                                            encoding="utf-8")
    return {**entry, "dmn": f"{ASIDE}/{slot.name}.dmn",
            "rule": {"statement": answer["statement"], "reason": answer["reason"], "owner": slot.owner,
                     "version": source["edition"] or "", "governs": list(slot.governs),
                     "cites": [given["document"]]}}


def _checked(env: Environment, entry: dict, failures: list[str], attempts: int) -> dict:
    digest = derived.dmn_digest(env.path.resolve().parent, entry)
    return {**entry, "status": "proposed" if failures else "checked", "attempts": attempts,
            "check": {"passed": not failures, "failures": failures,
                      "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "by": RUN,
                      **({"dmn_sha256": digest} if digest else {})}}


def _attempt(env: Environment, slot: Slot, extractor: Extractor, prompt: str, on: date) -> tuple[dict, dict, list]:
    """One answer from the model, as an entry, with the check's failures."""
    try:
        answer = extractor.extract(SYSTEM, prompt, answer_schema(slot))
    except AnswerError as error:
        return {}, _blank(slot), [f"answer: {error}"]
    try:
        entry = _entry(env, slot, answer)
        return answer, entry, derive_check.check(env, slot.kind, slot.name, entry, on=on)
    except (KeyError, TypeError, AttributeError) as error:
        return answer, _blank(slot), [f"answer: not in the schema: {_first_line(error)}"]


def derive_slot(env: Environment, slot: Slot, extractor: Extractor, on: date) -> dict:
    """One slot derived and checked: asked once, and once more with the failures if the check fails."""
    documents = candidates(env, slot, on)
    if not documents:
        return _checked(env, _blank(slot), [f"source: no document in {slot.collection} in force on {on} matches"], 0)
    prompt = _prompt(slot, documents, on)
    answer, entry, failures = _attempt(env, slot, extractor, prompt, on)
    if not failures:
        return _checked(env, entry, failures, 1)
    again = (f"{prompt}\n\nYour answer was:\n{json.dumps(answer)}\n\nIt failed the check:\n"
             + "\n".join(f"- {failure}" for failure in failures) + "\n\nAnswer again, fixing every failure.")
    _, entry, failures = _attempt(env, slot, extractor, again, on)
    return _checked(env, entry, failures, 2)


def _read_again(env: Environment, slot: Slot, entry: dict, reader: Extractor, on: date) -> dict:
    """The second reader's answer compared with the entry: {model, agreed, differences}; raises DeriveError when it
    cannot be made."""
    if ("value" if slot.kind == "facts" else "dmn") not in entry:
        raise DeriveError("the first reading gave nothing to compare")
    ident = str((entry.get("source") or {}).get("document") or "")
    document = env.store.document(slot.collection, ident, None) if env.store and ident else None
    if not document or not in_force(document, on):
        raise DeriveError(f"{slot.collection} holds no document {ident or '(none cited)'} in force on {on}")
    answer = reader.extract(SYSTEM, _prompt(slot, [document], on), answer_schema(slot))
    try:
        if slot.kind == "facts":
            agreed = derive_check.same_value(entry.get("value"), answer["value"])
            differences = [] if agreed else [{"first": entry.get("value"), "second": answer["value"]}]
        else:
            first = load_table(env.path.resolve().parent / "derived" / str(entry.get("dmn") or ""))
            with tempfile.TemporaryDirectory() as folder:
                file = Path(folder) / f"{slot.name}.dmn"
                file.write_text(render_dmn(slot, answer["rows"], ident), encoding="utf-8")
                again = load_table(file)
            differences = derive_check.table_differences(first, again, list(slot.outputs), slot.inputs)
    except (KeyError, TypeError, AttributeError, DmnError) as error:
        raise DeriveError(f"its answer cannot be compared: {_first_line(error)}") from error
    return {"model": reader.model, "agreed": not differences, "differences": differences[:derive_check.KEPT]}


def second_reading(env: Environment, slot: Slot, entry: dict, reader: Extractor, on: date) -> tuple[dict, list[str]]:
    """A second model derives the entry again from its cited document, never seeing the first answer, and the two
    are compared: a decision over the grid of inputs, a fact within 0.5%. -> the record for check.second_reading,
    and the failure lines: a disagreement, or a reading that could not be made, which never counts as agreement."""
    try:
        record = _read_again(env, slot, entry, reader, on)
    except DeriveError as error:
        text = _first_line(error)
        return ({"model": reader.model, "agreed": False, "differences": [], "error": text},
                [f"a second reading by {reader.model} could not be made: {text}"])
    if record["agreed"]:
        return record, []
    return record, [derive_check.disagreement(reader.model, record["differences"])]


def make_extractor(backend: str, model: str | None = None, ollama_url: str = OLLAMA_URL) -> Extractor:
    """Claude or a local model through Ollama, as clm-agent reaches them."""
    if backend == "ollama":
        return OllamaExtractor(model or LOCAL_MODEL, ollama_url)
    try:
        import anthropic  # pylint: disable=import-outside-toplevel
    except ImportError as missing:
        raise DeriveError(f"Claude needs the agent extra ({missing.name or 'anthropic'} is missing): pip install "
                          "'clm-decision-agent[agent]', or use --backend ollama") from missing
    return ClaudeExtractor(anthropic.Anthropic(), model or MODEL)


def reader_for(env: Environment) -> Extractor | None:
    """The second reader the environment's [derived] second_reader names, or None; through Ollama, the one its
    [documents] ollama_url names, which embeds its passages."""
    reader = env.second_reader
    if not reader:
        return None
    with open(env.path, "rb") as file:
        settings = document_settings(tomllib.load(file))
    return make_extractor(reader["backend"], reader["model"], settings.ollama_url if settings else OLLAMA_URL)


def _family(model: str) -> str:
    """A model's family: case and spaces ignored; for Claude the id without a trailing date, for an Ollama model the
    name before its tag, so any tag of gemma4 (gemma4:26b, gemma4:latest, gemma4) is gemma4."""
    name = model.strip().lower()
    return re.sub(r"-\d{8}$", "", name) if name.startswith("claude") else name.split(":", 1)[0]


def refuse_same_model(reader: Extractor | None, model: str | None) -> None:
    """Raise DeriveError when the second reader is of the family of the model that derived the layer: another tag
    or dated id may be the same weights, and would repeat its own reading."""
    if reader is not None and model and _family(reader.model) == _family(model):
        raise DeriveError(f"the second reader {reader.model.strip()} is the model that derived the layer, {model}, "
                          "or another tag of it, and would repeat its own reading: name another in [derived] "
                          "second_reader")


def _read_twice(env: Environment, slot: Slot, entry: dict, reader: Extractor, on: date) -> dict:
    """run's entry with the second reading's failures joined to its check."""
    record, failures = second_reading(env, slot, entry, reader, on)
    check = entry["check"]
    failures = [*check["failures"], *failures]
    return {**entry, "status": "proposed" if failures else "checked",
            "check": {**check, "passed": not failures, "failures": failures, "second_reading": record}}


def _unchanged(env: Environment, slot: Slot, entry: dict, on: date) -> bool:
    """Whether the entry's cited document is still the edition in force."""
    source = entry.get("source") or {}
    document = env.store.document(slot.collection, str(source.get("document", "")), None)
    return bool(document) and in_force(document, on) and edition(document) == source.get("edition")


def _replace(folder: Path, layer: dict) -> None:
    """Point each entry whose table the run wrote aside at derived/rules/, write the layer aside too, then move the
    tables into derived/rules/, replacing the ones there, and the layer straight after them."""
    rules = folder / "derived" / RULES
    rules.mkdir(parents=True, exist_ok=True)
    moves = []
    for kind in ("decisions", "facts"):
        for entry in layer[kind].values():
            if str(entry.get("dmn") or "").startswith(f"{ASIDE}/"):
                name = entry["dmn"].removeprefix(f"{ASIDE}/")
                moves.append((folder / "derived" / entry["dmn"], rules / name))
                entry["dmn"] = f"{RULES}/{name}"
    partial = derived.write_aside(folder, layer)
    for source, target in moves:
        os.replace(source, target)
    os.replace(partial, derived.path(folder))


def run(path: str | Path, collection: str, extractor: Extractor, on: date, *, fresh: bool = False) -> dict:
    """Derive every slot in the collection that is not approved or rejected on an unchanged edition, or every slot
    in it with `fresh`, which also drops the entries of slots no longer declared, and write the layer with its
    conflicts; other collections' entries are kept. The tables are written aside and replace the old ones only once
    every slot is derived, so a run that stops leaves the tables and the layer as they were. Raises DeriveError, writing nothing, when no slot is in the collection.
    -> {slot: status}, for the collection's slots."""
    env = load_environment(path, today=on)
    if not any(slot.collection == collection for slot in env.slots.values()):
        names = ", ".join(repr(name) for name in sorted({s.collection for s in env.slots.values()}))
        raise DeriveError(f"no slot in [derived] is in collection {collection!r}; "
                          + (f"the slots are in {names}" if names else "the environment declares no [derived] slots"))
    reader = reader_for(env)
    refuse_same_model(reader, extractor.model)
    folder = env.path.resolve().parent
    try:
        layer = _derive_all(env, collection, extractor, reader, on, fresh)
        _replace(folder, layer)
    finally:
        shutil.rmtree(folder / "derived" / ASIDE, ignore_errors=True)
        derived.path(folder).with_suffix(".json.partial").unlink(missing_ok=True)
    return {name: layer[slot.kind][name]["status"] for name, slot in env.slots.items()
            if slot.collection == collection and name in layer[slot.kind]}


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def _derive_all(env: Environment, collection: str, extractor: Extractor, reader: Extractor | None, on: date,
                fresh: bool) -> dict:
    """The layer run writes: each slot derived (and read again with `reader`) unless kept, with its conflicts."""
    folder = env.path.resolve().parent
    old = derived.read(folder)
    if fresh:   # the collection's entries and those of slots gone are dropped; other collections' are kept
        old = {**old, **{kind: {name: entry for name, entry in old[kind].items()
                                if name in env.slots and env.slots[name].collection != collection}
                         for kind in ("decisions", "facts")}}
    layer = {**old, "collection": collection, "model": extractor.model,
             "derived_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for name, slot in env.slots.items():
        kept = old[slot.kind].get(name)
        if slot.collection != collection or (kept and kept.get("status") in ("approved", "rejected")
                                             and _unchanged(env, slot, kept, on)):
            continue
        entry = derive_slot(env, slot, extractor, on)
        layer[slot.kind][name] = _read_twice(env, slot, entry, reader, on) if reader else entry
    found = derive_check.conflicts(env, layer)
    for name, conflict in found.items():
        before = old["conflicts"].get(name) or {}
        same = (before.get("supplied"), before.get("derived")) == (conflict["supplied"], conflict["derived"])
        layer["conflicts"][name] = before if same else conflict
    layer["conflicts"] = {name: item for name, item in layer["conflicts"].items() if name in found}
    return layer


def _extractor(args) -> Extractor:
    return make_extractor(args.backend, args.model, args.ollama_url)


def _day(text: str | None) -> date:
    try:
        return date.fromisoformat(text) if text else date.today()
    except ValueError as error:
        raise DeriveError(f"--as-of must be a date such as 2026-12-01, not {text!r}") from error


def _check(environment: str, folder: Path, on: date) -> int:
    env = load_environment(environment, today=on)
    reader = reader_for(env)
    refuse_same_model(reader, derived.read(folder).get("model"))

    def second(kind: str, name: str, entry: dict) -> tuple[dict, list[str]]:
        slot = env.slots.get(name)
        if slot is None or slot.kind != kind:
            error = f"the environment has no {kind} slot {name}"
            return ({"model": reader.model, "agreed": False, "differences": [], "error": error},
                    [f"a second reading by {reader.model} could not be made: {error}"])
        return second_reading(env, slot, entry, reader, on)
    counts = derive_check.recheck(folder, env, on=on, second=second if reader else None)
    print(", ".join(f"{counts[status]} {status}" for status in ("checked", "proposed", "stale", "untouched")))
    if counts["undigested"]:
        print(f"{counts['undigested']} approved before table digests were kept: sent back to proposed and checked, to "
              "be approved again")
    print(f"second reading by {reader.model}: {counts['agreed']} agreed, {counts['disagreed']} disagreed, "
          f"{counts['not_made']} not made" if reader else "no second reading: [derived] names no second_reader")
    return 0


def _command(args) -> int:
    folder = Path(args.environment).resolve().parent
    if args.command == "status":
        reader = second_reader_of(args.environment)
        entry = derived.set_status(folder, args.kind, args.name, args.status, by=args.by, note=args.note,
                                   second_reader=reader["model"] if reader else None)
        print(f"{args.kind} {args.name}: {entry['status']}, by {entry['decided_by']}")
        return 0
    on = _day(args.as_of)
    if args.command == "check":
        return _check(args.environment, folder, on)
    extractor = _extractor(args)
    before = derived.read(folder)
    statuses = run(args.environment, args.collection, extractor, on, fresh=args.fresh)
    after = derived.read(folder)
    kept = {name for name in statuses
            if any(before[kind].get(name) is not None and before[kind].get(name) == after[kind].get(name)
                   for kind in ("decisions", "facts"))}
    for name, status in statuses.items():
        print(f"{name}: {status}{', kept' if name in kept else ''}")
    if statuses and kept == set(statuses):
        print("Nothing derived: every slot is approved or rejected on an unchanged edition (--fresh derives them "
              "again)")
    print(f"written to {derived.path(folder)}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:  # pylint: disable=too-many-locals
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Derive an environment's rules and facts from its documents, "
                                                 "check them, and record a person's decision on each.")
    commands = parser.add_subparsers(dest="command", required=True)
    ran = commands.add_parser("run", help="derive the slots not approved or rejected, and check them")
    ran.add_argument("environment")
    ran.add_argument("collection")
    ran.add_argument("--backend", choices=("claude", "ollama"), default="claude")
    ran.add_argument("--model", help=f"default: {MODEL} with claude, {LOCAL_MODEL} with ollama")
    ran.add_argument("--ollama-url", default=OLLAMA_URL)
    ran.add_argument("--as-of", help="the date documents must be in force on (default: today)")
    ran.add_argument("--fresh", action="store_true",
                     help="derive every slot in the collection again, approved and rejected ones too, and drop the "
                     "entries of slots no longer declared; other collections' entries are kept")
    checked = commands.add_parser("check", help="check every proposed or checked entry again")
    checked.add_argument("environment")
    checked.add_argument("--as-of", help="the date documents must be in force on (default: today)")
    status = commands.add_parser("status", help="record a person's decision on an entry")
    status.add_argument("environment")
    status.add_argument("kind", choices=derived.KINDS)
    status.add_argument("name")
    status.add_argument("status", choices=derived.PERSON)
    status.add_argument("--by", required=True, help="who decided")
    status.add_argument("--note", help="why")
    args = parser.parse_args(argv)
    try:
        return _command(args)
    except (DeriveError, derived.DerivedError, ScenarioError, StepError, DmnError, OSError) as error:
        print(f"error: {_first_line(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
