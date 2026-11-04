"""Load an environment: the graph, named queries, decisions, calculations and scenarios one organisation decides with.

An environment is a folder with environment.toml, the graph's CSV files, DMN tables and the owner's Python module
(docs/environment.md). The owner writes every query, rule, calculation and lever; an agent can only choose among
them. A mistake in the file stops loading with one line naming the setting, as for scenarios.
"""
from __future__ import annotations

import importlib.util
import keyword
import os
import re
import sys
import threading
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Protocol

from . import derived
from . import documents as docs
from .documents import _table, _text  # shared with the [documents] settings
from .dmn import DmnError, Table, load_table
from .graph import Graph
from .scenario import CLM_BACKENDS, DEFAULT_CLM, ScenarioError, StepError

BACKENDS = ("builtin", "neo4j")
INFORMATION_KINDS = ("fact", "estimate", "supplied")
QUERY_KINDS = ("fact", "estimate")
TYPES = ("string", "number", "integer", "boolean", "list")
_NAME = re.compile(r"[a-z][a-z0-9_]*")
# The derived statuses environment.toml may admit, and those --admit-derived may, for testing; never rejected
FILE_ADMITS = ("approved", "checked")
OVERRIDE_ADMITS = ("proposed", "checked", "approved")
# A DMN input's typeRef, as a parameter type; "any" is not checked
_DMN_TYPES = {"number": "number", "integer": "number", "double": "number", "long": "number", "int": "number",
              "decimal": "number", "string": "string", "boolean": "boolean"}
# The scenario tools add this argument themselves
_RESERVED = ("branches",)


def _bad_name(name: str) -> str | None:
    """Why a parameter name cannot be a tool argument, or None."""
    if not _NAME.fullmatch(name):
        return "a name is lower case letters, digits and _"
    if keyword.iskeyword(name):
        return f"'{name}' is a Python keyword"
    return None


@dataclass(frozen=True)
class Query:
    """A named query the owner wrote: the built-in graph calls `function`; a later Cypher connector runs `cypher`."""
    name: str
    description: str
    params: dict[str, str]
    function: str | None
    cypher: str | None
    kind: str = "fact"


@dataclass(frozen=True)
class RuleInfo:  # pylint: disable=too-many-instance-attributes
    """What a person needs to know about a rule: what it says, why it exists, who owns it and which version applies;
    the kinds of decision it governs, the rules it depends on and the information it needs."""
    statement: str
    reason: str
    owner: str
    version: str
    governs: tuple[str, ...]
    depends_on: tuple[str, ...]
    needs: tuple[str, ...]
    cites: tuple[str, ...] = ()         # ids of documents in the decision's collection: where the rule comes from


@dataclass(frozen=True)
class Information:
    """A piece of information a rule needs: a recorded fact or an estimate a query provides, or something the person
    asking must supply."""
    name: str
    description: str
    kind: str
    query: str | None
    unit: str | None


@dataclass(frozen=True)
class Decision:  # pylint: disable=too-many-instance-attributes
    """A rule: a DMN table, or a function the owner wrote, called with the inputs."""
    name: str
    description: str
    params: dict[str, str]
    table: Table | None
    function: str | None
    rule: RuleInfo | None = None
    collection: str | None = None       # the collection its citations are in
    origin: str = "supplied"            # or "derived": read from the documents by a model, and checked
    status: str | None = None           # a derived rule's status in the derived layer
    source: dict | None = None          # a derived rule's source: document, passages, quote and edition


@dataclass(frozen=True)
class Slot:  # pylint: disable=too-many-instance-attributes
    """A decision or fact the environment needs and does not supply: derivation fills it from the documents. A
    decision has inputs and outputs; a fact names a node, a property and its type."""
    name: str
    kind: str                           # "decisions" or "facts"
    description: str
    collection: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    owner: str = ""
    governs: tuple[str, ...] = ()
    node: str | None = None
    property: str | None = None
    type: str | None = None


@dataclass(frozen=True)
class Calculation:
    """Arithmetic the owner wrote, such as adding up rows or a margin, traced like a rule."""
    name: str
    description: str
    params: dict[str, str]
    function: str


@dataclass(frozen=True)
class ScenarioDef:  # pylint: disable=too-many-instance-attributes
    """Alternatives to compare: the levers and their allowed values, the branch function, objective and limits."""
    name: str
    description: str
    params: dict[str, str]
    levers: dict[str, tuple]
    branch: str
    objective: str
    maximise: bool
    limits: tuple[str, ...]
    uncertain: tuple[str, ...] = ()     # estimate queries whose low and high are tried, and headroom searched
    approvals: tuple[str, ...] = ()     # decisions saying whether someone must approve: listed, never rejecting


@dataclass(frozen=True)
class Judge:
    """A yes/no question CLM answers about a text, such as a piece of news; the answer is a probability."""
    name: str
    description: str
    instructions: str


class Clm:
    """CLM for an environment's judges, loaded on the first judgement, so an environment that never judges never
    loads it. Calls take turns: CLM runs one model."""

    def __init__(self, backend: str, model_id: str, device: str | None):
        self.backend, self.model_id, self.device = backend, model_id, device
        self._engine = None
        self._lock = threading.Lock()

    def probability(self, judge: Judge, text: str) -> float:
        """CLM's probability that the judge's question is answered yes for the text; raises StepError if CLM cannot
        load."""
        with self._lock:
            if self._engine is None:
                # Loading can fail in many ways (no clm extra, no MLX, no network for the download, no memory)
                try:
                    from . import clm_router  # pylint: disable=import-outside-toplevel
                    self._engine = clm_router.load_engine(self.backend, self.model_id, None, self.device)
                except Exception as error:  # pylint: disable=broad-exception-caught
                    # Download errors run to several lines; the first says what failed
                    first = (str(error).strip().splitlines() or [type(error).__name__])[0]
                    raise StepError(f"judge {judge.name}: CLM could not load: {first}") from error
            question = {"type": "noul", "instructions": judge.instructions}
            return float(self._engine.answer(text, {judge.name: question})["answers"][judge.name]["noul"])


class Connector(Protocol):  # pylint: disable=too-few-public-methods
    """Where named queries run: the built-in graph now, a graph database later."""

    def run(self, query: Query, params: dict) -> list[dict]:
        """The query's rows."""


class BuiltinConnector:  # pylint: disable=too-few-public-methods
    """Runs a query by calling the owner's function with the built-in graph."""

    def __init__(self, graph: Graph, functions: ModuleType):
        self.graph, self.functions = graph, functions

    def run(self, query: Query, params: dict) -> list[dict]:
        """The rows the query's function returns."""
        return getattr(self.functions, query.function)(self.graph, **params)

    def run_recording(self, query: Query, params: dict) -> tuple[list[dict], set[str]]:
        """The rows, and the ids of every node the function was given: what it may have read. A node it was given
        and ignored is counted too, so this can over-report, never under-report."""
        recording = _Recording(self.graph)
        return getattr(self.functions, query.function)(recording, **params), recording.read


class _Recording:  # pylint: disable=too-few-public-methods
    """The graph, noting the id of each node it hands out; one per query call, so calls never share it."""

    def __init__(self, graph: Graph):
        self._graph, self.read = graph, set()

    def _note(self, found):
        for item in found if isinstance(found, list) else [found]:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                self.read.add(item["id"])
        return found

    def __getattr__(self, name: str):
        method = getattr(self._graph, name)
        return lambda *args, **kwargs: self._note(method(*args, **kwargs))


@dataclass(frozen=True)
class Environment:  # pylint: disable=too-many-instance-attributes
    """One organisation's decisions, ready to run. `graph` is the built-in graph, or None on Neo4j."""
    path: Path
    name: str
    description: str
    graph: Graph | None
    functions: ModuleType
    connector: Connector
    queries: dict[str, Query]
    decisions: dict[str, Decision]
    calculations: dict[str, Calculation]
    scenarios: dict[str, ScenarioDef]
    judges: dict[str, Judge]
    clm: Clm | None
    information: dict[str, Information] = field(default_factory=dict)
    collections: dict[str, docs.Collection] = field(default_factory=dict)
    store: docs.DocumentStore | None = None
    embedder: docs.Embedder | None = None
    document_tools: bool = True
    cited_text: bool = True
    slots: dict[str, Slot] = field(default_factory=dict)
    unfilled: dict[str, str] = field(default_factory=dict)          # slot name -> why it is not admitted
    admitted: tuple[str, ...] = ()                                  # the slots a derived entry fills
    admit: tuple[str, ...] = ()                                     # the statuses admitted
    admit_override: bool = False                                    # --admit-derived: for testing only
    derived_facts: dict[str, dict[str, dict]] = field(default_factory=dict)    # node -> property -> entry
    second_reader: dict | None = None       # [derived] second_reader: {backend, model}, or None for no second reading
    slot_statuses: dict[str, str] = field(default_factory=dict)     # slot name -> its entry's status, or not derived

    def topics(self) -> list[str]:
        """The kinds of decision the rules govern."""
        return sorted({topic for decision in self.decisions.values() if decision.rule
                       for topic in decision.rule.governs})


def _read(path: Path) -> dict:
    try:
        with open(path, "rb") as file:
            return tomllib.load(file)
    except OSError as error:
        raise ScenarioError(f"{path}: cannot read: {error.strerror}") from error
    except tomllib.TOMLDecodeError as error:
        raise ScenarioError(f"{path.name}: not valid TOML: {error}") from error


def is_environment(path: str | Path) -> bool:
    """Whether the file is an environment (it has an [environment] table) rather than a CLM scenario."""
    return isinstance(_read(Path(path)).get("environment"), dict)


def _params(table: dict, where: str) -> dict[str, str]:
    params = _table(table, "params", where)
    for name, kind in params.items():
        problem = _bad_name(name) or (f"'{name}' is reserved" if where.startswith("scenarios.") and name in _RESERVED
                                      else None)
        if problem:
            raise ScenarioError(f"{where}.params.{name}: {problem}")
        if kind not in TYPES:
            raise ScenarioError(f"{where}.params.{name}: type must be {', '.join(TYPES)}, not {kind!r}")
    return dict(params)


def _function(table: dict, where: str, functions: ModuleType) -> str:
    name = _text(table, "function", where)
    if not callable(getattr(functions, name, None)):
        raise ScenarioError(f"{where}.function: '{name}' is not a function in {functions.__name__}")
    return name


def _parts(data: dict, key: str) -> dict[str, dict]:
    parts = _table(data, key, key)
    for name, table in parts.items():
        if not _NAME.fullmatch(name):
            raise ScenarioError(f"{key}.{name}: a name is lower case letters, digits and _")
        if not isinstance(table, dict):
            raise ScenarioError(f"{key}.{name}: must be a table")
    return parts


def _import(folder: Path, name: str) -> ModuleType:
    """The owner's module, loaded afresh from its file, so each environment gets its own copy."""
    path = folder / f"{name}.py"
    # Appended, not inserted first, so a file beside the environment cannot shadow a framework module
    if str(folder) not in sys.path:
        sys.path.append(str(folder))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise ScenarioError(f"environment.functions: cannot import '{name}': {type(error).__name__}: {error}") \
            from error
    return module


@dataclass(frozen=True)
class GraphSettings:
    """The [graph] table: the backend, the CSV files, and where Neo4j is."""
    backend: str
    nodes: Path | None
    edges: Path | None
    uri: str
    user: str
    password_env: str
    database: str


def _graph_settings(data: dict, folder: Path, backend: str | None) -> GraphSettings:
    table = _table(data, "graph", "graph")
    backend = backend or table.get("backend", "builtin")
    if backend not in BACKENDS:
        raise ScenarioError(f"graph.backend: must be {' or '.join(BACKENDS)}, not {backend!r}")
    files = [folder / _text(table, key, "graph") if key in table or backend == "builtin" else None
             for key in ("nodes", "edges")]
    neo4j = _table(table, "neo4j", "graph.neo4j")
    return GraphSettings(backend, *files, _text(neo4j, "uri", "graph.neo4j", "bolt://127.0.0.1:7687"),
                         _text(neo4j, "user", "graph.neo4j", "neo4j"),
                         _text(neo4j, "password_env", "graph.neo4j", "NEO4J_PASSWORD"),
                         _text(neo4j, "database", "graph.neo4j", "neo4j"))


def graph_settings(path: str | Path) -> GraphSettings:
    """The environment's [graph] settings, for load_graph.py; raises ScenarioError naming the setting."""
    path = Path(path)
    return _graph_settings(_read(path), path.resolve().parent, None)


def load_graph_files(settings: GraphSettings) -> Graph:
    """The built-in graph from the settings' CSV files; raises ScenarioError naming the file."""
    if settings.nodes is None or settings.edges is None:
        raise ScenarioError("graph: needs nodes and edges, the CSV files of the graph")
    try:
        return Graph.from_csv(settings.nodes, settings.edges)
    except (OSError, ValueError) as error:
        raise ScenarioError(f"graph: {error}") from error


def neo4j_password(settings: GraphSettings) -> str:
    """The password from the variable the settings name; never from the file itself."""
    password = os.environ.get(settings.password_env, "")
    if not password:
        raise ScenarioError(f"graph.neo4j.password_env: set {settings.password_env} to the database's password")
    return password


def _connector(settings: GraphSettings, graph: Graph | None, functions: ModuleType) -> Connector:
    if settings.backend == "builtin":
        return BuiltinConnector(graph, functions)
    from .graph_neo4j import Neo4jConnector, connect  # pylint: disable=import-outside-toplevel
    driver = connect(settings.uri, settings.user, neo4j_password(settings))
    return Neo4jConnector(driver, settings.uri, settings.database)


def _clm(data: dict) -> Clm | None:
    """CLM's settings, as a scenario has them; None without a [clm] table."""
    if "clm" not in data:
        return None
    table = _table(data, "clm", "clm")
    if "heads" in table:
        raise ScenarioError("clm.heads: not supported for judges yet: they use CLM's released heads")
    backend = table.get("backend", "mlx")
    if backend not in CLM_BACKENDS:
        raise ScenarioError(f"clm.backend: must be {' or '.join(CLM_BACKENDS)}, not {backend!r}")
    device = table.get("device")
    if device is not None and backend != "torch":
        raise ScenarioError("clm.device: applies to the torch backend only")
    return Clm(backend, _text(table, "model_id", "clm", DEFAULT_CLM[backend]), device)


def _judge(name: str, table: dict, clm: Clm | None) -> Judge:
    where = f"judges.{name}"
    if clm is None:
        raise ScenarioError(f"{where}: a judge needs CLM: add a [clm] table (docs/environment.md)")
    if table.get("type", "noul") != "noul":
        raise ScenarioError(f"{where}.type: only noul (a yes/no question) for now")
    return Judge(name, _text(table, "description", where), _text(table, "instructions", where))


def _query(name: str, table: dict, functions: ModuleType, backend: str) -> Query:
    where = f"queries.{name}"
    function = _function(table, where, functions) if "function" in table or backend == "builtin" else None
    cypher = table.get("cypher")
    if backend == "neo4j" and (not isinstance(cypher, str) or not cypher.strip()):
        raise ScenarioError(f"{where}.cypher: the neo4j backend runs a query's cypher; write one")
    kind = table.get("kind", "fact")
    if kind not in QUERY_KINDS:
        raise ScenarioError(f"{where}.kind: must be {' or '.join(QUERY_KINDS)}, not {kind!r}")
    return Query(name, _text(table, "description", where), _params(table, where), function, cypher, kind)


def _names(table: dict, key: str, where: str) -> tuple[str, ...]:
    value = table.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ScenarioError(f"{where}.{key}: must be a list of names")
    return tuple(value)


def _rule(table: dict, where: str) -> RuleInfo | None:
    if "rule" not in table:
        return None
    rule = _table(table, "rule", where)
    where = f"{where}.rule"
    return RuleInfo(*(_text(rule, key, where) for key in ("statement", "reason", "owner", "version")),
                    *(_names(rule, key, where) for key in ("governs", "depends_on", "needs", "cites")))


def _information(name: str, table: dict, queries: dict[str, Query]) -> Information:
    where = f"information.{name}"
    kind = table.get("kind")
    if kind not in INFORMATION_KINDS:
        raise ScenarioError(f"{where}.kind: must be {', '.join(INFORMATION_KINDS)}, not {kind!r}")
    query = table.get("query")
    if kind == "supplied":
        if query is not None:
            raise ScenarioError(f"{where}.query: supplied information comes from the person asking, not a query")
    else:
        query = _text(table, "query", where)
        if query not in queries:
            raise ScenarioError(f"{where}.query: unknown query '{query}'")
        if queries[query].kind != kind:
            raise ScenarioError(f"{where}.kind: '{kind}', but query {query} gives a {queries[query].kind}")
    unit = _text(table, "unit", where) if "unit" in table else None
    return Information(name, _text(table, "description", where), kind, query, unit)


def _check_rules(decisions: dict[str, Decision], information: dict[str, Information]) -> None:
    """Every rule's dependencies are rules and its needs are information; no rule depends on itself."""
    for name, decision in decisions.items():
        if decision.rule is None:
            continue
        where = f"decisions.{name}.rule"
        for other in decision.rule.depends_on:
            if other not in decisions or decisions[other].rule is None:
                raise ScenarioError(f"{where}.depends_on: '{other}' is not a decision with a rule")
        for item in decision.rule.needs:
            if item not in information:
                raise ScenarioError(f"{where}.needs: unknown information '{item}'")
    for name in decisions:
        path, todo = [], [(name, 0)]
        while todo:
            current, depth = todo.pop()
            del path[depth:]
            if current in path:
                raise ScenarioError(f"decisions.{name}.rule.depends_on: a cycle: {' -> '.join(path + [current])}")
            path.append(current)
            rule = decisions[current].rule
            todo.extend((other, depth + 1) for other in (rule.depends_on if rule else ()))


def _decision(name: str, table: dict, folder: Path, functions: ModuleType) -> Decision:
    where = f"decisions.{name}"
    description = _text(table, "description", where)
    if ("dmn" in table) == ("function" in table):
        raise ScenarioError(f"{where}: needs either dmn or function")
    rule = _rule(table, where)
    collection = _text(table, "collection", where) if "collection" in table else None
    if "function" in table:
        return Decision(name, description, _params(table, where), None, _function(table, where, functions), rule,
                        collection)
    path = folder / _text(table, "dmn", where)
    if not path.is_file():
        raise ScenarioError(f"{where}.dmn: no file {path.name}")
    try:
        dmn = load_table(path)
    except DmnError as error:
        raise ScenarioError(f"{where}.dmn: {error}") from error
    for item in dmn.inputs:
        if _bad_name(item):
            raise ScenarioError(f"{where}.dmn: input '{item}' cannot be a parameter: {_bad_name(item)}")
    params = {item: _DMN_TYPES.get(kind.rsplit(":", 1)[-1], "any") for item, kind in zip(dmn.inputs, dmn.types)}
    return Decision(name, description, params, dmn, None, rule, collection)


def _citations(decision: Decision) -> list[tuple[str, str]]:
    """Each id a decision cites, with where: its rule's cites, then its table's citation cells (ids split by ;)."""
    where = f"decisions.{decision.name}"
    found = [(f"{where}.rule.cites", cited) for cited in (decision.rule.cites if decision.rule else ())]
    table = decision.table
    if table is not None and "citation" in table.outputs:
        column = table.outputs.index("citation")
        for row in table.rules:
            cell = row.values[column]
            found += [(f"{where}.dmn: row {row.id}", cited.strip())
                      for cited in (cell.split(";") if isinstance(cell, str) else []) if cited.strip()]
    return found


def _check_citations(decisions: dict[str, Decision], collections: dict, store) -> None:
    """Every id a rule cites must be in its decision's collection, so no rule cites what cannot be read."""
    for name, decision in decisions.items():
        where = f"decisions.{name}"
        cited = _citations(decision)
        if decision.collection is None:
            if cited:
                raise ScenarioError(f"{where}.collection: name the collection its citations are in")
            continue
        if decision.collection not in collections:
            raise ScenarioError(f"{where}.collection: no collection {decision.collection} "
                                f"({', '.join(sorted(collections)) or 'none declared'})")
        for place, ident in cited:
            if not store.exists(decision.collection, ident):
                if place.endswith(".rule.cites"):
                    raise ScenarioError(f"{place}: {decision.collection} holds no document {ident}")
                raise ScenarioError(f"{place} cites {ident}, which {decision.collection} does not hold")


def _scenario(name: str, table: dict, functions: ModuleType, decisions, queries: dict) -> ScenarioDef:  # pylint: disable=too-many-locals
    where = f"scenarios.{name}"
    levers = _table(table, "levers", where)
    if not levers:
        raise ScenarioError(f"{where}.levers: needs at least one lever")
    for lever, values in levers.items():
        if not isinstance(values, list) or not values:
            raise ScenarioError(f"{where}.levers.{lever}: needs at least one value, as a list")
    limits = table.get("limits", [])
    if not isinstance(limits, list) or not all(isinstance(limit, str) for limit in limits):
        raise ScenarioError(f"{where}.limits: must be a list of decision names")
    for limit in limits:
        if limit not in decisions:
            raise ScenarioError(f"{where}.limits: unknown decision '{limit}'")
    branch = _text(table, "branch", where)
    if not callable(getattr(functions, branch, None)):
        raise ScenarioError(f"{where}.branch: '{branch}' is not a function in {functions.__name__}")
    maximise = table.get("maximise", True)
    if not isinstance(maximise, bool):
        raise ScenarioError(f"{where}.maximise: must be true or false")
    uncertain = _names(table, "uncertain", where)
    for query in uncertain:
        if query not in queries or queries[query].kind != "estimate":
            raise ScenarioError(f"{where}.uncertain: '{query}' is not an estimate query")
    approvals = _names(table, "approvals", where)
    for decision in approvals:
        if decision not in decisions:
            raise ScenarioError(f"{where}.approvals: unknown decision '{decision}'")
    return ScenarioDef(name, _text(table, "description", where), _params(table, where),
                       {lever: tuple(values) for lever, values in levers.items()},
                       branch, _text(table, "objective", where), maximise, tuple(limits), uncertain, approvals)


def in_force(document: dict, on: date) -> bool:
    """Whether a document, as the store returns it, is in force on the date: effective_from on or before it and
    effective_until, if any, on or after it; a document with no dates always is."""
    metadata = document.get("metadata") or {}
    start, end = metadata.get("effective_from"), metadata.get("effective_until")
    return (not start or str(start) <= on.isoformat()) and (not end or on.isoformat() <= str(end))


def edition(document: dict) -> str | None:
    """A document's edition: the date it took effect, or None when it has none."""
    start = (document.get("metadata") or {}).get("effective_from")
    return str(start) if start else None


def _slot_names(table: dict, where: str) -> dict[str, str]:
    value = _table(table, where.rsplit(".", 1)[-1], where)
    for name, kind in value.items():
        if _bad_name(name):
            raise ScenarioError(f"{where}.{name}: {_bad_name(name)}")
        if kind not in TYPES:
            raise ScenarioError(f"{where}.{name}: type must be {', '.join(TYPES)}, not {kind!r}")
    return dict(value)


def _slots(data: dict, backend: str, decisions: dict) -> tuple[dict[str, Slot], tuple[str, ...]] | None:
    """The [derived] table's slots and the statuses it admits; None without the table, when the layer is ignored."""
    if "derived" not in data:
        return None
    table = _table(data, "derived", "derived")
    admit = _names(table, "admit", "derived") if "admit" in table else ("approved",)
    for status in admit:
        if status not in FILE_ADMITS:
            raise ScenarioError(f"derived.admit: {status!r} cannot be admitted here: only {' or '.join(FILE_ADMITS)}; "
                                "a proposed entry is admitted only by --admit-derived, for testing, and a rejected one "
                                "never")
    default = table.get("collection")
    slots = {}
    for name, part in _table(table, "decisions", "derived.decisions").items():
        where = f"derived.decisions.{name}"
        if _bad_name(name) or not isinstance(part, dict):
            raise ScenarioError(f"{where}: a name is lower case letters, digits and _, and its value a table")
        if name in decisions:
            raise ScenarioError(f"{where}: a supplied decision has this name; a slot is for one not supplied")
        slots[name] = Slot(name, "decisions", _text(part, "description", where),
                           _text(part, "collection", where, default), _slot_names(part, f"{where}.inputs"),
                           _slot_names(part, f"{where}.outputs"), _text(part, "owner", where),
                           _names(part, "governs", where))
    for node, properties in _table(table, "facts", "derived.facts").items():
        for prop, part in (properties.items() if isinstance(properties, dict) else ()):
            where = f"derived.facts.{node}.{prop}"
            if backend != "builtin":
                raise ScenarioError(f"{where}: derived facts need the built-in graph, not {backend}")
            if not isinstance(part, dict) or part.get("type") not in TYPES:
                raise ScenarioError(f"{where}.type: must be {', '.join(TYPES)}")
            slots[f"{node}.{prop}"] = Slot(f"{node}.{prop}", "facts", _text(part, "description", where),
                                           _text(part, "collection", where, default), node=node, property=prop,
                                           type=part["type"])
    return slots, admit


READER_BACKENDS = ("claude", "ollama")


def _second_reader(data: dict) -> dict | None:
    """The [derived] table's second_reader, the model that reads each derived entry's document again; None without
    one."""
    table = data.get("derived") or {}
    if "second_reader" not in table:
        return None
    where, reader = "derived.second_reader", table["second_reader"]
    if not isinstance(reader, dict):
        raise ScenarioError(f"{where}: must be a table such as {{ backend = \"ollama\", model = \"gemma4:26b\" }}")
    extra = sorted(set(reader) - {"backend", "model"})
    if extra:
        raise ScenarioError(f"{where}: unknown setting {', '.join(extra)}; only backend and model")
    if reader.get("backend") not in READER_BACKENDS:
        raise ScenarioError(f"{where}.backend: must be {' or '.join(READER_BACKENDS)}")
    if not isinstance(reader.get("model"), str) or not reader["model"].strip():
        raise ScenarioError(f"{where}.model: must name the model, such as gemma4:26b")
    return {"backend": reader["backend"], "model": reader["model"].strip()}


def second_reader_of(path: str | Path) -> dict | None:
    """The second reader an environment file's [derived] table names, read without loading the environment (no
    store is connected); None without one. Raises ScenarioError as loading would."""
    data = _read(Path(path))
    return _second_reader(data) if "derived" in data else None


def _why_not(entry: dict | None, admit: tuple[str, ...], override: bool, store, slot: Slot, on: date,
             folder: Path) -> str | None:
    """Why a derived entry may not fill its slot, or None when it may."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-return-statements
    if entry is None:
        return "nothing has been derived for it: run clm-derive run"
    status = entry.get("status")
    if status not in admit:
        return f"its status is {status}, and only {', '.join(admit)} entries are admitted"
    passed = (entry.get("check") or {}).get("passed") is True
    if not passed and not (override and "proposed" in admit):
        return "it failed its check, or was never checked: run clm-derive check"
    if derived.changed_since_check(folder, entry):
        return "its DMN has changed since its check: run clm-derive check again"
    source = entry.get("source") or {}
    document = store.document(slot.collection, source.get("document", ""), None) if store else {}
    if not document:
        return f"its source {source.get('document')} is not in {slot.collection}"
    if not in_force(document, on):
        return f"its source {source.get('document')} is not in force on {on}"
    if edition(document) != source.get("edition"):
        return (f"it was derived from the {source.get('edition')} edition of {source.get('document')}, and the "
                f"edition in force is {edition(document)}: run clm-derive run")
    return None


def _derived_decision(slot: Slot, entry: dict, folder: Path, functions: ModuleType, collections: dict,
                      store) -> Decision:
    """A derived entry as a decision; raises ScenarioError when its table does not load, is not the slot's, or
    cites what its collection does not hold."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    table = {key: entry[key] for key in ("description", "dmn", "rule") if key in entry}
    table["collection"] = entry.get("collection") or slot.collection
    decision = _decision(slot.name, table, folder / "derived", functions)
    if decision.params != slot.inputs:
        raise ScenarioError(f"its table's inputs {decision.params} are not the slot's {slot.inputs}")
    decision = Decision(decision.name, decision.description, decision.params, decision.table, None, decision.rule,
                        decision.collection, "derived", entry.get("status"), entry.get("source"))
    _check_citations({slot.name: decision}, collections, store)
    return decision


@dataclass
class _Derived:
    """What the derived layer added: decisions, facts set on the graph, and the slots left unfilled."""
    decisions: dict[str, Decision] = field(default_factory=dict)
    facts: dict[str, dict[str, dict]] = field(default_factory=dict)
    unfilled: dict[str, str] = field(default_factory=dict)
    admitted: list[str] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)


NOT_DERIVED = "not derived"     # the status shown for a slot the layer has no entry for


def _fill(slots: dict[str, Slot], context: tuple, graph: Graph | None, collections: dict, store) -> _Derived:
    """Fill each slot from the layer where its entry may be admitted: a fact only where the supplied graph has no
    value (supplied decides), a decision only when its table loads and its citations resolve."""
    # pylint: disable=too-many-locals
    folder, functions, admit, override, on = context
    layer = derived.read(folder)
    found = _Derived()
    for name, slot in slots.items():
        entry = layer[slot.kind].get(name)
        found.statuses[name] = entry.get("status") if entry is not None else NOT_DERIVED
        why = _why_not(entry, admit, override, store, slot, on, folder)
        if why is None and slot.kind == "facts":
            supplied = graph.node(slot.node) if slot.node in {n["id"] for n in graph.nodes()} else None
            if supplied is None:
                why = f"the graph has no node {slot.node}"
            elif slot.property in supplied:
                continue        # the supplied value stands; clm-derive reports a conflict when they differ
            elif not _CHECK_TYPE[slot.type](entry.get("value")):
                why = f"its value {entry.get('value')!r} is not of type {slot.type}"
            else:
                graph.set(slot.node, slot.property, entry["value"])
                found.facts.setdefault(slot.node, {})[slot.property] = entry
        elif why is None:
            try:
                found.decisions[name] = _derived_decision(slot, entry, folder, functions, collections, store)
            except ScenarioError as error:
                why = f"its table cannot be used: {error}"
        if why is None:
            found.admitted.append(name)
        else:
            found.unfilled[name] = why
    return found


_CHECK_TYPE = {"string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
               "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
               "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
               "list": lambda v: isinstance(v, list)}


# pylint: disable-next=too-many-locals,too-many-arguments
def load_environment(path: str | Path, backend: str | None = None,
                     store: docs.DocumentStore | None = None,
                     embedder: docs.Embedder | None = None, *, admit: list[str] | None = None,
                     today: date | None = None) -> Environment:
    """Read and check an environment file; raises ScenarioError naming the setting. `backend` overrides the file's
    [graph] backend, to run the same environment on the other graph. `store` and `embedder` replace the document
    store and embedder the file names, as tests do. With a [derived] table, admitted derived rules and facts join
    the supplied ones (docs/environment.md); `admit` overrides the statuses admitted, for testing only, and `today`
    is the date a cited document must be in force on."""
    path = Path(path)
    data = _read(path)
    folder = path.resolve().parent
    head = _table(data, "environment", "environment")
    functions = _import(folder, _text(head, "functions", "environment"))
    settings = _graph_settings(data, folder, backend)
    graph = load_graph_files(settings) if settings.backend == "builtin" else None
    declared = _slots(data, settings.backend, _parts(data, "decisions"))
    slots, admitted_statuses = declared or ({}, ())
    if admit is not None and declared is not None:
        for status in admit:
            if status not in OVERRIDE_ADMITS:
                raise ScenarioError(f"--admit-derived: {status!r} cannot be admitted: only "
                                    f"{', '.join(OVERRIDE_ADMITS)}; a rejected entry never decides")
        admitted_statuses = tuple(admit)
    queries = {name: _query(name, table, functions, settings.backend)
               for name, table in _parts(data, "queries").items()}
    decisions = {name: _decision(name, table, folder, functions) for name, table in _parts(data, "decisions").items()}
    information = {name: _information(name, table, queries) for name, table in _parts(data, "information").items()}
    _check_rules(decisions, information)
    calculations = {name: Calculation(name, _text(table, "description", f"calculations.{name}"),
                                      _params(table, f"calculations.{name}"),
                                      _function(table, f"calculations.{name}", functions))
                    for name, table in _parts(data, "calculations").items()}
    scenarios = {name: _scenario(name, table, functions, {*decisions, *slots}, queries)
                 for name, table in _parts(data, "scenarios").items()}
    clm = _clm(data)
    judges = {name: _judge(name, table, clm) for name, table in _parts(data, "judges").items()}
    document_settings = docs.document_settings(data)
    collections = docs.collections(data, document_settings)
    if document_settings is not None:
        store = store or docs.connect(document_settings)
        embedder = embedder or docs.OllamaEmbedder(document_settings.ollama_url)
    else:
        store = embedder = None
    _check_citations(decisions, collections, store)
    filled = _fill(slots, (folder, functions, admitted_statuses, admit is not None, today or date.today()), graph,
                   collections, store)
    # Connected last, once everything else is known to be right
    return Environment(path, _text(head, "name", "environment"), _text(head, "description", "environment"), graph,
                       functions, _connector(settings, graph, functions), queries, decisions | filled.decisions,
                       calculations, scenarios,
                       judges, clm, information, collections, store, embedder,
                       document_settings.tools if document_settings else True,
                       document_settings.cited_text if document_settings else True, slots, filled.unfilled,
                       tuple(filled.admitted), admitted_statuses, admit is not None and declared is not None,
                       filled.facts, _second_reader(data) if declared is not None else None, filled.statuses)
