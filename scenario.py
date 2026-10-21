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
CLM_BACKENDS = ("mlx", "torch")
CLM_DEVICES = ("cuda", "mps", "cpu")
# Settings of the chat assistant this framework used to be; a scenario that still has them is out of date
RETIRED_ROUTE_KEYS = ("system_prompt", "tools", "fields")
DEFAULT_CLM = {"mlx": "mlx-community/CLM-v0.1-8B-MLX-8bit", "torch": "Contrastive-LM/CLM-v0.1-8B"}
DEFAULT_GENERATOR = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_TOOL_CHECK = "Is this tool call appropriate for what the user asked?"
# Completes "Write N different ..." in synthesize.py's request to the generator
DEFAULT_SYNTHESIZE = "messages that a person might send to an assistant"


class ScenarioError(ValueError):
    """The scenario file is missing, unreadable or invalid."""


_RULE = re.compile(r"\s*(\w+)\s*(>=|<=|==)\s*(.+?)\s*")


@dataclass(frozen=True)
class Rule:
    """Guidance for the caller that applies when one of CLM's answers meets a condition."""

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
    """One kind of request: what it means to CLM (description) and the tools that inform its decisions."""

    name: str
    description: str
    examples: tuple[str, ...] = ()
    inform: tuple[str, ...] = ()


@dataclass(frozen=True)
class Inform:
    """An informing tool's settings: the questions its facts are given to (None: all of them)."""

    tool: str
    questions: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Action:
    """An acting tool, run after a decision on one of `routes` when `when` holds (always, if None)."""

    tool: str
    routes: tuple[str, ...]
    when: Rule | None = None
    confirm: bool = False


@dataclass(frozen=True)
class Scenario:  # pylint: disable=too-many-instance-attributes
    """Everything read from one scenario file."""

    path: Path
    sha: str
    clm_model_id: str
    generator_backend: str
    generator_model: str
    max_tokens: int
    routing_instructions: str
    confidence_threshold: float
    routes: dict[str, Route]
    questions: dict[str, dict]
    tool_check_instructions: str
    tool_check_threshold: float
    live_corrections: bool
    match_threshold: float
    rules: tuple[Rule, ...] = ()
    tools_module: str | None = None
    tool_settings: dict = field(default_factory=dict)
    clm_heads: Path | None = None
    clm_backend: str = "mlx"
    clm_device: str | None = None
    generator_base_url: str | None = None
    generator_key_env: str | None = None
    record_file: Path | None = None
    record_key_env: str | None = None
    record_key_file: Path | None = None
    synthesize_messages: str = DEFAULT_SYNTHESIZE
    informs: dict[str, Inform] = field(default_factory=dict)
    actions: tuple[Action, ...] = ()
    # Informing tools run before routing, whose facts the route hears too
    route_inform: tuple[str, ...] = ()

    @property
    def record_path(self) -> Path:
        """The record of route corrections that belongs to this scenario file."""
        return self.record_file or self.path.with_name(self.path.stem + ".corrections.jsonl")

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


def _names(table: dict, key: str, where: str, what: str) -> list[str]:
    value = table.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ScenarioError(f"{where}.{key} must be a list of {what}")
    return value


def _route(name: str, table) -> Route:
    where = f"routes.{name}"
    if not isinstance(table, dict):
        raise ScenarioError(f"[{where}] must be a table")
    for key in RETIRED_ROUTE_KEYS:
        if key in table:
            raise ScenarioError(f"{where}.{key} is no longer used; see docs/scenario.md")
    return Route(name, _text(table, "description", where),
                 tuple(_names(table, "examples", where, "example messages")),
                 tuple(_names(table, "inform", where, "tool names")))


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
    return _condition(_text(table, "when", where), where, questions, _text(table, "tell", where))


def _condition(when: str, where: str, questions: dict[str, dict], tell: str = "") -> Rule:
    """A `when` condition, as a Rule (whose `tell` is empty when the condition belongs to an action)."""
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


def _tools(table: dict, questions: dict[str, dict]) -> tuple[str | None, dict, dict[str, Inform]]:
    module = table.get("module")
    if module is not None and (not isinstance(module, str) or not module.strip()):
        raise ScenarioError("tools.module must be a non-empty string")
    if "confirm" in table:
        raise ScenarioError("tools.confirm is no longer used; an action's confirm setting replaces it")
    settings = table.get("settings", {})
    if not isinstance(settings, dict):
        raise ScenarioError("[tools.settings] must be a table")
    informs = {}
    for name, inform in _table(table, "inform").items():
        where = f"tools.inform.{name}"
        if not isinstance(inform, dict):
            raise ScenarioError(f"[{where}] must be a table")
        if "questions" not in inform:
            informs[name] = Inform(name)
            continue
        names = _names(inform, "questions", where, "question names")
        for question in names:
            if question not in questions:
                raise ScenarioError(f"{where}.questions: unknown question '{question}'")
        informs[name] = Inform(name, tuple(names))
    return module, dict(settings), informs


def _action(index: int, table, routes: dict[str, Route], questions: dict[str, dict]) -> Action:
    where = f"actions[{index}]"
    if not isinstance(table, dict):
        raise ScenarioError(f"{where} must be a table")
    tool = _text(table, "tool", where)
    names = table.get("routes")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ScenarioError(f"{where}.routes must be a non-empty list of route names")
    for name in names:
        if name not in routes:
            raise ScenarioError(f"{where}.routes: unknown route '{name}'")
    when = _condition(_text(table, "when", where), where, questions) if "when" in table else None
    confirm = table.get("confirm", False)
    if not isinstance(confirm, bool):
        raise ScenarioError(f"{where}.confirm must be true or false")
    if confirm:
        raise ScenarioError(f"{where}.confirm: only false is supported for now")
    return Action(tool, tuple(names), when, confirm)


def _clm_backend(clm: dict) -> tuple[str, str | None]:
    """-> (backend, device) from [clm]; the settings of the retired clm-serve backend are errors."""
    backend = clm.get("backend", "mlx")
    if backend == "server":
        raise ScenarioError('clm.backend "server" is no longer supported; see docs/models.md')
    for key in ("url", "api_key_env"):
        if key in clm:
            raise ScenarioError(f"clm.{key} is no longer used; see docs/models.md")
    if backend not in CLM_BACKENDS:
        raise ScenarioError(f"clm.backend must be one of {', '.join(CLM_BACKENDS)}")
    device = clm.get("device")
    if device is not None:
        if backend != "torch":
            raise ScenarioError("clm.device applies to the torch backend only")
        if device not in CLM_DEVICES:
            raise ScenarioError(f"clm.device must be one of {', '.join(CLM_DEVICES)}")
    return backend, device


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
    clm_backend, clm_device = _clm_backend(clm)
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
    if "catalogue" in data:
        raise ScenarioError("[catalogue] is no longer used; an informing tool takes its place")
    tools_module, tool_settings, informs = _tools(_table(data, "tools"), questions)
    action_tables = data.get("actions", [])
    if not isinstance(action_tables, list):
        raise ScenarioError("actions must be written as [[actions]] tables")

    return Scenario(
        path=path,
        sha=hashlib.sha256(raw).hexdigest(),
        clm_model_id=_text(clm, "model_id", "clm", DEFAULT_CLM[clm_backend]),
        clm_backend=clm_backend,
        clm_device=clm_device,
        generator_base_url=_address(generator, "base_url", "generator") if backend == "openai" else None,
        generator_key_env=_optional(generator, "api_key_env", "generator"),
        record_file=_beside(path, record, "path", "record"),
        record_key_env=_optional(record, "key_env", "record"),
        record_key_file=_beside(path, record, "key_file", "record"),
        generator_backend=backend,
        generator_model=_text(generator, "model", "generator", DEFAULT_GENERATOR),
        max_tokens=_number(generator, "max_tokens", "generator", 400, 1, integer=True),
        routing_instructions=_text(routing, "instructions", "routing"),
        confidence_threshold=_number(routing, "confidence_threshold", "routing", 0.6, 0, 1),
        route_inform=tuple(_names(routing, "inform", "routing", "tool names")),
        routes=routes,
        questions=questions,
        tool_check_instructions=_text(tool_check, "instructions", "tool_check", DEFAULT_TOOL_CHECK),
        tool_check_threshold=_number(tool_check, "threshold", "tool_check", 0.5, 0, 1),
        live_corrections=live,
        match_threshold=_number(record, "match_threshold", "record", 0.97, 0, 1),
        rules=tuple(_rule(index, table, questions) for index, table in enumerate(rule_tables)),
        tools_module=tools_module,
        tool_settings=tool_settings,
        clm_heads=path.parent / _text(clm, "heads", "clm") if "heads" in clm else None,
        synthesize_messages=_text(_table(data, "synthesize"), "messages", "synthesize", DEFAULT_SYNTHESIZE),
        informs=informs,
        actions=tuple(_action(index, table, routes, questions) for index, table in enumerate(action_tables)),
    )


def load_for_command(path: str | Path) -> Scenario | None:
    """For command-line tools: the scenario, or None after printing why it could not be loaded."""
    try:
        return load_scenario(path)
    except ScenarioError as error:
        print(f"error: {error}", file=sys.stderr)
        return None
