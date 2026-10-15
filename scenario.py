"""Load and validate scenario.toml."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

QUESTION_TYPES = ("noul", "choice", "score")
BACKENDS = ("mlx", "ollama", "openai")
CLM_BACKENDS = ("mlx", "server")
DEFAULT_SERVER_MODEL = "clm-latest"
# Routes become graph node names, so they cannot collide with the built-in nodes
RESERVED_ROUTE_NAMES = frozenset({
    "assess", "clarify", "tools", "verify_tool", "decline_tool",
    "review_route", "review_tool", "review_reply", "settle", "confirm_tool", "__start__", "__end__",
})
DEFAULT_CLM = "mlx-community/CLM-v0.1-8B-MLX-4bit"
DEFAULT_GENERATOR = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_TOOL_CHECK = "Is this tool call appropriate for what the user asked?"
# Completes "Write N different ..." in synthesize.py's request to the generator
DEFAULT_SYNTHESIZE = "messages that a person might send to an assistant"


class ScenarioError(ValueError):
    """The scenario file is missing, unreadable or invalid."""


_RULE = re.compile(r"\s*(\w+)\s*(>=|<=|==)\s*(.+?)\s*")


@dataclass(frozen=True)
class Rule:
    """Guidance for the generator that applies when one of CLM's answers meets a condition."""

    when: str
    question: str
    operator: str
    value: float | str
    tell: str

    def applies(self, answers: dict) -> bool:
        """Whether CLM's answer to this rule's question meets the condition."""
        answer = answers.get(self.question)
        if answer is None:
            return False
        if answer["type"] == "choice":
            return answer["choice"] == self.value
        actual = answer["noul"] if answer["type"] == "noul" else answer["score"]
        return actual >= self.value if self.operator == ">=" else actual <= self.value


@dataclass(frozen=True)
class Route:
    """One destination: what it means to CLM (description) and how the generator should answer."""

    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Catalogue:
    """A context tool whose entries CLM chooses between on some routes, filling one intake field."""

    tool: str
    routes: tuple[str, ...]
    field: str
    instructions: str
    confidence_threshold: float
    questions: tuple[str, ...] = ()    # the scenario's questions asked again with the picked entry's facts


@dataclass(frozen=True)
class Scenario:  # pylint: disable=too-many-instance-attributes
    """Everything read from one scenario file."""

    path: Path
    sha: str
    clm_model_id: str
    history_turns: int
    generator_backend: str
    generator_model: str
    max_tokens: int
    routing_instructions: str
    confidence_threshold: float
    routes: dict[str, Route]
    questions: dict[str, dict]
    tool_check_instructions: str
    tool_check_threshold: float
    max_tool_rounds: int
    live_corrections: bool
    match_threshold: float
    rules: tuple[Rule, ...] = ()
    tools_module: str | None = None
    confirm_tools: tuple[str, ...] = ()
    tool_settings: dict = field(default_factory=dict)
    clm_heads: Path | None = None
    clm_backend: str = "mlx"
    clm_url: str | None = None
    clm_key_env: str | None = None
    generator_base_url: str | None = None
    generator_key_env: str | None = None
    record_file: Path | None = None
    record_key_env: str | None = None
    record_key_file: Path | None = None
    synthesize_messages: str = DEFAULT_SYNTHESIZE
    catalogue: Catalogue | None = None

    @property
    def record_path(self) -> Path:
        """The review record that belongs to this scenario file."""
        return self.record_file or self.path.with_name(self.path.stem + ".record.jsonl")

    @property
    def routing_sha(self) -> str:
        """A fingerprint of the routing wording. Trained heads only fit the wording they were trained on."""
        wording = [self.routing_instructions, [[name, route.description] for name, route in self.routes.items()]]
        return hashlib.sha256(json.dumps(wording).encode("utf-8")).hexdigest()

    def matching_rules(self, answers: dict) -> list[Rule]:
        """The rules whose condition CLM's answers meet, in file order."""
        return [rule for rule in self.rules if rule.applies(answers)]


def _table(data: dict, key: str) -> dict:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ScenarioError(f"[{key}] must be a table")
    return value


def _text(table: dict, key: str, where: str, default: str | None = None) -> str:
    value = table.get(key, default)
    if value is None:
        raise ScenarioError(f"{where}: missing '{key}'")
    if not isinstance(value, str) or not value.strip():
        raise ScenarioError(f"{where}.{key} must be a non-empty string")
    return value


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _optional(table: dict, key: str, where: str) -> str | None:
    return _text(table, key, where) if key in table else None


def _beside(scenario_path: Path, table: dict, key: str, where: str) -> Path | None:
    """A path setting, taken as relative to the scenario file's folder."""
    if key not in table:
        return None
    try:
        return scenario_path.parent / Path(_text(table, key, where)).expanduser()
    except RuntimeError as error:       # "~someone" for a user that does not exist
        raise ScenarioError(f"{where}.{key}: {error}") from error


def _address(table: dict, key: str, where: str) -> str:
    """A server address. Credentials belong in an api_key_env variable, where they are not printed."""
    url = _text(table, key, where)
    if not url.startswith(("http://", "https://")):
        raise ScenarioError(f"{where}.{key} must start with http:// or https://")
    if "@" in url.split("//", 1)[1].split("/", 1)[0] or "\\" in url:
        raise ScenarioError(f"{where}.{key} must not contain a user name or password; use api_key_env")
    return url


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _number(table: dict, key: str, where: str, default, low, high=None, integer: bool = False):
    value = table.get(key, default)
    kinds = int if integer else (int, float)
    in_range = isinstance(value, kinds) and not isinstance(value, bool) and value >= low \
        and (high is None or value <= high)
    if not in_range:
        bound = f"from {low} to {high}" if high is not None else f"of at least {low}"
        raise ScenarioError(f"{where}.{key} must be {'an integer' if integer else 'a number'} {bound}")
    return value


def _route(name: str, table) -> Route:
    where = f"routes.{name}"
    if name in RESERVED_ROUTE_NAMES:
        raise ScenarioError(f"route name '{name}' is reserved; choose another name")
    if not isinstance(table, dict):
        raise ScenarioError(f"[{where}] must be a table")
    tools = table.get("tools", [])
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise ScenarioError(f"{where}.tools must be a list of tool names")
    examples = table.get("examples", [])
    if not isinstance(examples, list) or not all(isinstance(e, str) for e in examples):
        raise ScenarioError(f"{where}.examples must be a list of example messages")
    fields = table.get("fields", {})
    if not isinstance(fields, dict):
        raise ScenarioError(f"[{where}.fields] must be a table")
    for key in fields:
        _text(fields, key, f"{where}.fields")
    return Route(name, _text(table, "description", where), _text(table, "system_prompt", where),
                 tuple(tools), tuple(examples), dict(fields))


def _question(name: str, table) -> dict:
    where = f"questions.{name}"
    if not isinstance(table, dict):
        raise ScenarioError(f"[{where}] must be a table")
    if table.get("type") not in QUESTION_TYPES:
        raise ScenarioError(f"{where}.type must be one of {', '.join(QUESTION_TYPES)}")
    _text(table, "instructions", where)
    criteria = table.get("criteria")
    if table["type"] == "choice" and not (isinstance(criteria, dict) and len(criteria) >= 2):
        raise ScenarioError(f"{where}.criteria must be a table of at least two options")
    if table["type"] == "score" and not (isinstance(criteria, list) and len(criteria) >= 2):
        raise ScenarioError(f"{where}.criteria must be a list of at least two levels")
    return dict(table)


def _rule(index: int, table, questions: dict[str, dict]) -> Rule:
    where = f"rules[{index}]"
    if not isinstance(table, dict):
        raise ScenarioError(f"{where} must be a table")
    when, tell = _text(table, "when", where), _text(table, "tell", where)
    match = _RULE.fullmatch(when)
    if not match:
        raise ScenarioError(f"{where}.when must look like '<question> >= <number>' or '<question> == <option>'")
    name, operator, raw = match.groups()
    if name not in questions:
        raise ScenarioError(f"{where}.when: unknown question '{name}'")
    question = questions[name]
    if question["type"] == "choice":
        options = list(question["criteria"])
        value = raw.strip("'\"")
        if operator != "==" or value not in options:
            raise ScenarioError(f"{where}.when: use == with one of {', '.join(options)} for question '{name}'")
        return Rule(when, name, operator, value, tell)
    try:
        number = float(raw)
    except ValueError:
        number = None
    if operator == "==" or number is None:
        raise ScenarioError(f"{where}.when: use >= or <= with a number for question '{name}'")
    return Rule(when, name, operator, number, tell)


def _tools(table: dict) -> tuple[str | None, tuple[str, ...], dict]:
    module = table.get("module")
    if module is not None and (not isinstance(module, str) or not module.strip()):
        raise ScenarioError("tools.module must be a non-empty string")
    confirm = table.get("confirm", [])
    if not isinstance(confirm, list) or not all(isinstance(name, str) for name in confirm):
        raise ScenarioError("tools.confirm must be a list of tool names")
    settings = table.get("settings", {})
    if not isinstance(settings, dict):
        raise ScenarioError("[tools.settings] must be a table")
    return module, tuple(confirm), dict(settings)


def _catalogue(data: dict, routes: dict[str, Route], questions: dict[str, dict]) -> Catalogue | None:
    if "catalogue" not in data:
        return None
    table, where = _table(data, "catalogue"), "catalogue"
    names = table.get("routes")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ScenarioError("catalogue.routes must be a non-empty list of route names")
    tool, field_name = _text(table, "tool", where), _text(table, "field", where)
    for name in names:
        if name not in routes:
            raise ScenarioError(f"catalogue.routes: unknown route '{name}'")
        if field_name not in routes[name].fields:
            raise ScenarioError(f"catalogue.field: route '{name}' has no field '{field_name}'")
    informed = table.get("questions", list(questions))
    if not isinstance(informed, list) or not all(isinstance(name, str) for name in informed):
        raise ScenarioError("catalogue.questions must be a list of question names")
    for name in informed:
        if name not in questions:
            raise ScenarioError(f"catalogue.questions: unknown question '{name}'")
    return Catalogue(tool, tuple(names), field_name, _text(table, "instructions", where),
                     _number(table, "confidence_threshold", where, 0.5, 0, 1), tuple(informed))


def load_scenario(path: str | Path) -> Scenario:  # pylint: disable=too-many-locals
    """Read and validate a scenario file; raises ScenarioError with the offending key."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise ScenarioError(f"scenario file not found: {path}") from None
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ScenarioError(f"{path}: invalid TOML: {error}") from None

    clm, generator, routing = _table(data, "clm"), _table(data, "generator"), _table(data, "routing")
    tool_check, record = _table(data, "tool_check"), _table(data, "record")

    backend = generator.get("backend", "mlx")
    if backend not in BACKENDS:
        raise ScenarioError(f"generator.backend must be one of {', '.join(BACKENDS)}")
    clm_backend = clm.get("backend", "mlx")
    if clm_backend not in CLM_BACKENDS:
        raise ScenarioError(f"clm.backend must be one of {', '.join(CLM_BACKENDS)}")
    served = clm_backend == "server"
    if "key_env" in record and "key_file" in record:
        raise ScenarioError("record: set key_env or key_file, not both")
    routes = {name: _route(name, table) for name, table in _table(data, "routes").items()}
    if len(routes) < 2:
        raise ScenarioError("a scenario needs at least two routes")
    live = record.get("live_corrections", True)
    if not isinstance(live, bool):
        raise ScenarioError("record.live_corrections must be true or false")
    questions = {name: _question(name, table) for name, table in _table(data, "questions").items()}
    rule_tables = data.get("rules", [])
    if not isinstance(rule_tables, list):
        raise ScenarioError("rules must be written as [[rules]] tables")
    tools_module, confirm_tools, tool_settings = _tools(_table(data, "tools"))
    for name, route in routes.items():
        # A confirm tool succeeding is what completes an intake; without one it would never close
        if route.fields and not set(route.tools) & set(confirm_tools):
            raise ScenarioError(f"routes.{name}: a route with fields needs one of its tools listed in "
                                f"tools.confirm, to submit the completed request")

    return Scenario(
        path=path,
        sha=hashlib.sha256(raw).hexdigest(),
        clm_model_id=_text(clm, "model", "clm", DEFAULT_SERVER_MODEL) if served
        else _text(clm, "model_id", "clm", DEFAULT_CLM),
        clm_backend=clm_backend,
        clm_url=_address(clm, "url", "clm").rstrip("/") if served else None,
        clm_key_env=_optional(clm, "api_key_env", "clm"),
        generator_base_url=_address(generator, "base_url", "generator") if backend == "openai" else None,
        generator_key_env=_optional(generator, "api_key_env", "generator"),
        record_file=_beside(path, record, "path", "record"),
        record_key_env=_optional(record, "key_env", "record"),
        record_key_file=_beside(path, record, "key_file", "record"),
        history_turns=_number(clm, "history_turns", "clm", 6, 1, integer=True),
        generator_backend=backend,
        generator_model=_text(generator, "model", "generator", DEFAULT_GENERATOR),
        max_tokens=_number(generator, "max_tokens", "generator", 400, 1, integer=True),
        routing_instructions=_text(routing, "instructions", "routing"),
        confidence_threshold=_number(routing, "confidence_threshold", "routing", 0.6, 0, 1),
        routes=routes,
        questions=questions,
        tool_check_instructions=_text(tool_check, "instructions", "tool_check", DEFAULT_TOOL_CHECK),
        tool_check_threshold=_number(tool_check, "threshold", "tool_check", 0.5, 0, 1),
        max_tool_rounds=_number(tool_check, "max_tool_rounds", "tool_check", 3, 0, integer=True),
        live_corrections=live,
        match_threshold=_number(record, "match_threshold", "record", 0.97, 0, 1),
        rules=tuple(_rule(index, table, questions) for index, table in enumerate(rule_tables)),
        tools_module=tools_module,
        confirm_tools=confirm_tools,
        tool_settings=tool_settings,
        clm_heads=path.parent / _text(clm, "heads", "clm") if "heads" in clm else None,
        synthesize_messages=_text(_table(data, "synthesize"), "messages", "synthesize", DEFAULT_SYNTHESIZE),
        catalogue=_catalogue(data, routes, questions),
    )


def load_for_command(path: str | Path) -> Scenario | None:
    """For command-line tools: the scenario, or None after printing why it could not be loaded."""
    try:
        return load_scenario(path)
    except ScenarioError as error:
        print(f"error: {error}", file=sys.stderr)
        return None
