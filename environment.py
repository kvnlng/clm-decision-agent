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
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

from dmn import DmnError, Table, load_table
from graph import Graph
from scenario import ScenarioError

BACKENDS = ("builtin", "neo4j")
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


@dataclass(frozen=True)
class Decision:
    """A rule: a DMN table, or a function the owner wrote, called with the inputs."""
    name: str
    description: str
    params: dict[str, str]
    table: Table | None
    function: str | None


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
    from graph_neo4j import Neo4jConnector, connect  # pylint: disable=import-outside-toplevel
    driver = connect(settings.uri, settings.user, neo4j_password(settings))
    return Neo4jConnector(driver, settings.uri, settings.database)


def _query(name: str, table: dict, functions: ModuleType, backend: str) -> Query:
    where = f"queries.{name}"
    function = _function(table, where, functions) if "function" in table or backend == "builtin" else None
    cypher = table.get("cypher")
    if backend == "neo4j" and (not isinstance(cypher, str) or not cypher.strip()):
        raise ScenarioError(f"{where}.cypher: the neo4j backend runs a query's cypher; write one")
    return Query(name, _text(table, "description", where), _params(table, where), function, cypher)


def _decision(name: str, table: dict, folder: Path, functions: ModuleType) -> Decision:
    where = f"decisions.{name}"
    description = _text(table, "description", where)
    if ("dmn" in table) == ("function" in table):
        raise ScenarioError(f"{where}: needs either dmn or function")
    if "function" in table:
        return Decision(name, description, _params(table, where), None, _function(table, where, functions))
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
    return Decision(name, description, params, dmn, None)


def _scenario(name: str, table: dict, functions: ModuleType, decisions: dict) -> ScenarioDef:
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
    return ScenarioDef(name, _text(table, "description", where), _params(table, where),
                       {lever: tuple(values) for lever, values in levers.items()},
                       branch, _text(table, "objective", where), maximise, tuple(limits))


def load_environment(path: str | Path, backend: str | None = None) -> Environment:
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
    calculations = {name: Calculation(name, _text(table, "description", f"calculations.{name}"),
                                      _params(table, f"calculations.{name}"),
                                      _function(table, f"calculations.{name}", functions))
                    for name, table in _parts(data, "calculations").items()}
    scenarios = {name: _scenario(name, table, functions, decisions)
                 for name, table in _parts(data, "scenarios").items()}
    # Connected last, once everything else is known to be right
    return Environment(path, _text(head, "name", "environment"), _text(head, "description", "environment"), graph,
                       functions, _connector(settings, graph, functions), queries, decisions, calculations, scenarios)
