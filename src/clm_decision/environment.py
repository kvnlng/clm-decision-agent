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
from pathlib import Path
from types import ModuleType
from typing import Protocol

from .dmn import DmnError, Table, load_table
from .graph import Graph
from .scenario import CLM_BACKENDS, DEFAULT_CLM, ScenarioError, StepError

BACKENDS = ("builtin", "neo4j")
INFORMATION_KINDS = ("fact", "estimate", "supplied")
QUERY_KINDS = ("fact", "estimate")
TYPES = ("string", "number", "integer", "boolean", "list")
_NAME = re.compile(r"[a-z][a-z0-9_]*")
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
class Decision:
    """A rule: a DMN table, or a function the owner wrote, called with the inputs."""
    name: str
    description: str
    params: dict[str, str]
    table: Table | None
    function: str | None
    rule: RuleInfo | None = None


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


def _table(data: dict, key: str, where: str) -> dict:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ScenarioError(f"{where}: must be a table")
    return value


def _text(table: dict, key: str, where: str, default: str | None = None) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ScenarioError(f"{where}.{key}: must be text")
    return value.strip()


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
                    *(_names(rule, key, where) for key in ("governs", "depends_on", "needs")))


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
    if "function" in table:
        return Decision(name, description, _params(table, where), None, _function(table, where, functions), rule)
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
    return Decision(name, description, params, dmn, None, rule)


def _scenario(name: str, table: dict, functions: ModuleType, decisions: dict, queries: dict) -> ScenarioDef:
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


def load_environment(path: str | Path, backend: str | None = None) -> Environment:  # pylint: disable=too-many-locals
    """Read and check an environment file; raises ScenarioError naming the setting. `backend` overrides the file's
    [graph] backend, to run the same environment on the other graph."""
    path = Path(path)
    data = _read(path)
    folder = path.resolve().parent
    head = _table(data, "environment", "environment")
    functions = _import(folder, _text(head, "functions", "environment"))
    settings = _graph_settings(data, folder, backend)
    graph = load_graph_files(settings) if settings.backend == "builtin" else None
    queries = {name: _query(name, table, functions, settings.backend)
               for name, table in _parts(data, "queries").items()}
    decisions = {name: _decision(name, table, folder, functions) for name, table in _parts(data, "decisions").items()}
    information = {name: _information(name, table, queries) for name, table in _parts(data, "information").items()}
    _check_rules(decisions, information)
    calculations = {name: Calculation(name, _text(table, "description", f"calculations.{name}"),
                                      _params(table, f"calculations.{name}"),
                                      _function(table, f"calculations.{name}", functions))
                    for name, table in _parts(data, "calculations").items()}
    scenarios = {name: _scenario(name, table, functions, decisions, queries)
                 for name, table in _parts(data, "scenarios").items()}
    clm = _clm(data)
    judges = {name: _judge(name, table, clm) for name, table in _parts(data, "judges").items()}
    # Connected last, once everything else is known to be right
    return Environment(path, _text(head, "name", "environment"), _text(head, "description", "environment"), graph,
                       functions, _connector(settings, graph, functions), queries, decisions, calculations, scenarios,
                       judges, clm, information)
