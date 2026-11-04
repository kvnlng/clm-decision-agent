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
from clm_decision.environment import graph_settings
from clm_decision.scenario import ScenarioError

DECISION_FIELDS = {"description", "dmn", "function", "params", "collection", "rule", "source", "status", "check",
                   "decided_by", "decided_at", "note"}


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


def read_layer(environment: Path, errors: list[str]) -> dict:
    """The derived layer, or an empty one with the reason in errors when it cannot be read."""
    try:
        return derived.read(Path(environment).parent)
    except (derived.DerivedError, ValueError, OSError) as error:
        errors.append(f"derived layer: {error}")
        return derived.empty(None)


def configuration(environment: Path, tools: list, described: dict | None, error: str | None) -> dict:
    """Everything the Configuration page shows: the environment's name and description, its server's tools, every
    rule (supplied and derived, with origin and status), the information items, the graph, the collections with when
    each was built, and every problem found reading them."""
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
        "tools": [{"name": tool.name, "description": tool.description or "",
                   "parameters": list((tool.input_schema or {}).get("properties", {}))} for tool in tools],
        "rules": rules,
        "information": [{"name": name, **table} for name, table in data.get("information", {}).items()],
        "graph": _graph(environment, errors) if "graph" in data else {"note": "no graph", "nodes": None,
                                                                      "edges": None},
        "collections": [{"name": name, "description": table.get("description", ""),
                         "built": (built.get(name) or {}).get("built")}
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
    node, _, prop = str(entry.get("name", name) if kind == "conflicts" else name).partition(".")
    return node, prop


def review(environment: Path) -> dict:  # pylint: disable=too-many-locals
    """The derived entries in the order a person reviews them: checked, open conflicts, proposed, then those decided.
    A fact or conflict carries the supplied value and the graph row it touches; a decision, its table's rows, its
    DMN text and the supplied rule of the same name, if any."""
    environment = Path(environment)
    errors: list[str] = []
    found, data = read_layer(environment, errors), _toml_or_empty(environment)
    nodes, folder = _nodes(environment, data), derived_folder(environment)
    items = []
    for kind in derived.KINDS:
        for name, entry in found[kind].items():
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
    """Approve, reject or resolve one entry, recording who and why; Refused when the layer does not allow it."""
    try:
        return derived.set_status(Path(environment).parent, kind, name, status, by=by, note=note or None)
    except derived.DerivedError as error:
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
    table; it goes back to proposed with no check. Refused, with nothing written, otherwise."""
    found, entry = _entry(environment, "decisions", name)
    folder = derived_folder(environment)
    if not entry.get("dmn"):
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
    os.replace(partial, path)
    return _proposed(environment, "decisions", name, found, by, note)
