"""The environment as the portal shows it, read from the files, and the changes it makes to the derived layer.

Nothing here loads the environment: that would import the owner's functions and connect to PostgreSQL in the
portal's own process. The environment file is read with tomllib, the graph's CSV files with csv, each decision table
with dmn.load_table and the derived layer with clm_decision.derived. Tools and collections' build times come from the
portal's server, when it is running.
"""
from __future__ import annotations

import csv
import os
import tomllib
from pathlib import Path

from clm_decision import derived, dmn
from clm_decision.environment import graph_settings, second_reader_of
from clm_decision.scenario import ScenarioError

DECISION_FIELDS = {"description", "dmn", "function", "params", "collection", "rule", "source", "status", "check",
                   "decided_by", "decided_at", "note"}
TEXT_FIELDS = ("description", "dmn", "function", "collection", "status", "decided_by", "decided_at", "note", "node",
               "property", "target", "name")
COLLECTION_TOOLS = ("search", "document", "cites", "cited_by")     # each collection's tools are KIND_COLLECTION


def read_toml(environment: Path) -> dict:
    """The environment file; raises ValueError naming it when it cannot be read."""
    try:
        return tomllib.loads(Path(environment).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"{Path(environment).name}: {error}") from error


def derived_folder(environment: Path) -> Path:
    """The derived layer's folder, beside the environment file."""
    return derived.path(Path(environment).parent).parent


def inside(path: Path, folder: Path) -> bool:
    """Whether the path, resolved, is inside the folder."""
    return path.resolve().is_relative_to(folder.resolve())


def table_rows(path: Path) -> dict:
    """A DMN table's inputs, outputs and rows as written; raises DmnError."""
    table = dmn.load_table(path)
    return {"inputs": table.inputs, "outputs": table.outputs, "hit_policy": table.hit_policy,
            "rows": [{"id": row.id, "when": list(row.entries), "then": list(row.values)} for row in table.rules]}


def _table(path: Path, errors: list[str], where: str) -> dict | None:
    try:
        return table_rows(path)
    except dmn.DmnError as error:
        errors.append(f"{where}: {error}")
        return None


def _supplied(name: str, table: dict, folder: Path, errors: list[str]) -> dict:
    found = {"name": name, "origin": "supplied", "status": None, "description": table.get("description", ""),
             "rule": table.get("rule") or {}, "collection": table.get("collection"), "dmn": table.get("dmn"),
             "function": table.get("function"), "source": None, "check": None, "table": None, "other": {}}
    if table.get("dmn"):
        found["table"] = _table(folder / table["dmn"], errors, f"decisions.{name}.dmn")
    return found


def _derived(name: str, entry: dict, folder: Path, errors: list[str]) -> dict:
    found = {"name": name, "origin": "derived", "status": entry.get("status"),
             "description": entry.get("description", ""), "rule": entry.get("rule") or {},
             "collection": entry.get("collection"), "dmn": entry.get("dmn"), "function": entry.get("function"),
             "source": entry.get("source"), "check": entry.get("check"), "decided_by": entry.get("decided_by"),
             "decided_at": entry.get("decided_at"), "note": entry.get("note"), "table": None,
             "other": {key: value for key, value in entry.items() if key not in DECISION_FIELDS}}
    if entry.get("dmn"):
        path = folder / entry["dmn"]
        if not inside(path, folder):
            errors.append(f"derived decisions.{name}.dmn: {entry['dmn']} is outside derived/, so it is not read")
        else:
            found["table"] = _table(path, errors, f"derived decisions.{name}.dmn")
    return found


def _csv(path: Path | None, errors: list[str]) -> dict | None:
    if path is None:
        return None
    try:
        with open(path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            return {"columns": list(reader.fieldnames or []), "rows": rows}
    except (OSError, csv.Error) as error:
        errors.append(f"graph: {path.name}: {error}")
        return None


def _graph(environment: Path, errors: list[str]) -> dict:
    try:
        settings = graph_settings(environment)
    except ScenarioError as error:
        errors.append(str(error))
        return {"note": None, "nodes": None, "edges": None}
    if settings.backend != "builtin":
        return {"note": "graph in Neo4j, not shown", "nodes": None, "edges": None}
    return {"note": None, "nodes": _csv(settings.nodes, errors), "edges": _csv(settings.edges, errors)}


def _texts(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _wrong(found: dict, key: str, where: str, what: str, errors: list[str], ok) -> None:
    if found.get(key) is not None and not ok(found[key]):
        errors.append(f"{where}.{key}: must be {what}, so it is not shown")
        found[key] = None


def clean(kind: str, name: str, entry, errors: list[str]) -> dict | None:
    """A copy of a derived entry for showing, with each wrongly typed field it relies on dropped and named in errors;
    None, with the reason, when the entry is not a table at all."""
    where = f"derived {kind}.{name}"
    if not isinstance(entry, dict):
        errors.append(f"{where}: not a table, so it is not shown")
        return None
    found = dict(entry)
    for key in TEXT_FIELDS:
        _wrong(found, key, where, "text", errors, lambda value: isinstance(value, str))
    for key in ("rule", "source"):
        _wrong(found, key, where, "a table", errors, lambda value: isinstance(value, dict))
    _wrong(found, "check", where, 'a table with "passed" true or false and "failures" a list of text', errors,
           lambda value: isinstance(value, dict) and isinstance(value.get("passed", False), bool)
           and _texts(value.get("failures", [])))
    if found.get("rule"):
        found["rule"] = dict(found["rule"])
        for key in ("governs", "depends_on", "needs", "cites"):
            _wrong(found["rule"], key, f"{where}.rule", "a list of text", errors, _texts)
    if found.get("source"):
        found["source"] = dict(found["source"])
        for key in ("document", "quote", "edition"):
            _wrong(found["source"], key, f"{where}.source", "text", errors, lambda value: isinstance(value, str))
        _wrong(found["source"], "passages", f"{where}.source", "a list of text", errors, _texts)
    return found


def read_layer(environment: Path, errors: list[str]) -> dict:
    """The derived layer for showing, each entry cleaned of wrongly typed fields; an empty one, with the reason in
    errors, when it cannot be read."""
    try:
        found = derived.read(Path(environment).parent)
    except (derived.DerivedError, ValueError, OSError) as error:
        errors.append(f"derived layer: {error}")
        return derived.empty(None)
    shown = {**found, **{kind: {} for kind in derived.KINDS}}
    _wrong(shown, "collection", "derived layer", "text", errors, lambda value: isinstance(value, str))
    for kind in derived.KINDS:
        for name, entry in found[kind].items():
            entry = clean(kind, name, entry, errors)
            if entry is not None:
                shown[kind][name] = entry
    return shown


def left_out(described: dict | None, collection: str) -> bool:
    """Whether the server's [serve] setting leaves out every tool of the collection: describe then drops it and
    names its tools in left_out."""
    if not described or collection in (described.get("collections") or {}):
        return False
    return any(f"{kind}_{collection}" in (described.get("left_out") or []) for kind in COLLECTION_TOOLS)


def tools_off(data: dict) -> bool:
    """Whether the environment file's [documents] tools is false: the server then offers no document tools, and
    describe lists no collection."""
    return (data.get("documents") or {}).get("tools", True) is False


def configuration(environment: Path, described: dict | None, error: str | None) -> dict:
    """Everything the Configuration page shows: the environment's name and description, every rule (supplied and
    derived, with origin and status), the information items, the graph, the collections with when each was built,
    and every problem found reading them. What the server offers over MCP is the MCP Inspector's to show."""
    environment = Path(environment)
    errors = [error] if error else []
    try:
        data = read_toml(environment)
    except ValueError as problem:
        errors.append(str(problem))
        data = {}
    folder = environment.parent
    about = data.get("environment", {})
    layer = read_layer(environment, errors)
    rules = [_supplied(name, table, folder, errors) for name, table in data.get("decisions", {}).items()]
    rules += [_derived(name, entry, derived_folder(environment), errors)
              for name, entry in layer["decisions"].items()]
    built = (described or {}).get("collections", {})
    return {
        "name": about.get("name", environment.parent.name), "description": about.get("description", ""),
        "rules": rules,
        "information": [{"name": name, **table} for name, table in data.get("information", {}).items()],
        "graph": _graph(environment, errors) if "graph" in data else {"note": "no graph", "nodes": None,
                                                                      "edges": None},
        "collections": [{"name": name, "description": table.get("description", ""),
                         "built": (built.get(name) or {}).get("built"), "left_out": left_out(described, name),
                         "tools_off": tools_off(data)}
                        for name, table in data.get("collections", {}).items()],
        "errors": errors,
    }


class Refused(ValueError):
    """A change the portal will not make; the message says why, and nothing was written."""


def _toml_or_empty(environment: Path) -> dict:
    try:
        return read_toml(environment)
    except ValueError:
        return {}


def _nodes(environment: Path, data: dict) -> dict[str, dict]:
    graph = _graph(environment, []) if "graph" in data else {"nodes": None}
    return {row.get("id"): row for row in (graph["nodes"] or {"rows": []})["rows"]}


def _target(kind: str, name: str, entry: dict) -> tuple[str, str]:
    """The graph node and property a fact, or a conflict over one, touches."""
    if kind == "facts" and entry.get("node") and entry.get("property"):
        return entry["node"], entry["property"]
    node, _, prop = str((entry.get("name") or name) if kind == "conflicts" else name).partition(".")
    return node, prop


def _slot_collection(data: dict, kind: str, name: str, entry: dict) -> str | None:
    """The collection environment.toml derives the entry's slot from: its own, or the [derived] table's."""
    table = data.get("derived") if isinstance(data.get("derived"), dict) else {}
    if kind == "decisions":
        part = (table.get("decisions") or {}).get(name)
    else:
        node, prop = _target(kind, name, entry)
        part = ((table.get("facts") or {}).get(node) or {}).get(prop)
    found = part.get("collection") if isinstance(part, dict) else None
    found = found or table.get("collection")
    return found if isinstance(found, str) else None


def review(environment: Path) -> dict:  # pylint: disable=too-many-locals
    """The derived entries in the order a person reviews them: checked, open conflicts, proposed, then those decided.
    A fact or conflict carries the supplied value and the graph row it touches; a decision, its table's rows, its
    DMN text and the supplied rule of the same name, if any. An entry that names no collection, as a fact derived
    before facts recorded one, takes its slot's, so its passage is looked up where it was derived."""
    environment = Path(environment)
    errors: list[str] = []
    found, data = read_layer(environment, errors), _toml_or_empty(environment)
    nodes, folder = _nodes(environment, data), derived_folder(environment)
    items = []
    for kind in derived.KINDS:
        for name, entry in found[kind].items():
            if not entry.get("collection") and _slot_collection(data, kind, name, entry):
                entry = {**entry, "collection": _slot_collection(data, kind, name, entry)}
            item = {"kind": kind, "name": name, "entry": entry, "status": entry.get("status"),
                    "supplied": None, "graph_row": None, "table": None, "dmn_text": None}
            if kind == "decisions":
                item["supplied"] = (data.get("decisions", {}).get(name) or {}).get("rule", {}).get("statement")
                path = folder / (entry.get("dmn") or "")
                if entry.get("dmn") and inside(path, folder) and path.is_file():
                    item["dmn_text"] = path.read_text(encoding="utf-8")
                    item["table"] = _table(path, errors, f"derived decisions.{name}.dmn")
            else:
                node, prop = _target(kind, name, entry)
                item["graph_row"] = nodes.get(node)
                item["supplied"] = entry.get("supplied") if kind == "conflicts" else (nodes.get(node) or {}).get(prop)
            items.append(item)
    order = (("Checked", ("checked",)), ("Open conflicts", ("open",)), ("Proposed", ("proposed",)),
             ("Decided", ("approved", "rejected", "resolved")))
    known = {status for _, statuses in order for status in statuses}
    groups = [(title, [item for item in items if item["status"] in statuses]) for title, statuses in order]
    groups.append(("Other", [item for item in items if item["status"] not in known]))
    return {"groups": [(title, group) for title, group in groups if group], "errors": errors}


# pylint: disable-next=too-many-arguments
def decide(environment: Path, kind: str, name: str, status: str, *, by: str, note: str | None) -> dict:
    """Approve, reject or resolve one entry, recording who and why; Refused when the layer does not allow it. With a
    [derived] second_reader, approving needs its agreeing reading, as clm-derive status does."""
    try:
        reader = second_reader_of(environment)
        return derived.set_status(Path(environment).parent, kind, name, status, by=by, note=note or None,
                                  second_reader=reader["model"] if reader else None)
    except (derived.DerivedError, ScenarioError) as error:
        raise Refused(str(error)) from error


def _entry(environment: Path, kind: str, name: str) -> tuple[dict, dict]:
    try:
        found = derived.read(Path(environment).parent)
    except (derived.DerivedError, ValueError, OSError) as error:
        raise Refused(f"the derived layer cannot be read: {error}") from error
    if name not in found[kind]:
        raise Refused(f"no {kind} entry named {name!r}")
    return found, found[kind][name]


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _proposed(environment: Path, kind: str, name: str, found: dict, by: str, note: str | None) -> dict:
    """A correction's last steps: in one write, the entry proposed with its check cleared; then the change recorded
    with who made it and why."""
    found[kind][name].update(status="proposed", check=None)
    derived.write(Path(environment).parent, found)
    return decide(environment, kind, name, "proposed", by=by, note=note)


def correct_fact(environment: Path, name: str, value, *, by: str, note: str | None) -> dict:
    """Give a derived fact a corrected value: it goes back to proposed with no check, so clm-derive check must pass
    it again before anyone can approve it."""
    found, entry = _entry(environment, "facts", name)
    entry["value"] = value
    return _proposed(environment, "facts", name, found, by, note)


def correct_decision(environment: Path, name: str, dmn_text: str, *, by: str, note: str | None) -> dict:
    """Replace a derived decision's DMN table with corrected text, which must stay inside derived/ and load as a
    table. The entry is written as proposed with no check before the table is replaced, so a failure part way never
    leaves a checked or approved entry with an unchecked table. Refused, with nothing written, otherwise."""
    found, entry = _entry(environment, "decisions", name)
    folder = derived_folder(environment)
    if not isinstance(entry.get("dmn"), str) or not entry["dmn"]:
        raise Refused(f"{name} has no DMN table to correct")
    path = folder / entry["dmn"]
    if not inside(path, folder) or path.suffix != ".dmn":
        raise Refused(f"{name}: {entry['dmn']} is outside derived/ or is not a .dmn file, so the portal will not "
                      "write it")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(dmn_text, encoding="utf-8")
    try:
        dmn.load_table(partial)
    except dmn.DmnError as error:
        partial.unlink()
        raise Refused(f"{name}: the corrected table cannot be used: {error}") from error
    try:
        changed = _proposed(environment, "decisions", name, found, by, note)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return changed
