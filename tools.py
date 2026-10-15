"""Load a scenario's tools from the Python module it names."""
from __future__ import annotations

import importlib
import sys

from langchain_core.tools import BaseTool

from scenario import Scenario, ScenarioError

MAX_ENTRIES = 50


def load_tools(scenario: Scenario) -> dict:
    """Tool name -> tool, from the module named in [tools]. The module's configure() is called if it has one."""
    name = scenario.tools_module
    if name is None:
        return {}
    scenario_dir = scenario.path.resolve().parent
    # Appended, not inserted first, so a file beside the scenario cannot shadow a framework module
    if str(scenario_dir) not in sys.path:
        sys.path.append(str(scenario_dir))
    # An adopter's module can fail in any way; each failure should be one line, not a traceback
    try:
        module = importlib.import_module(name)
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise ScenarioError(f"tools.module: cannot import '{name}': {type(error).__name__}: {error}") from error
    tools = getattr(module, "TOOLS", None)
    if not isinstance(tools, (list, tuple)) or not tools:
        raise ScenarioError(f"tools.module: '{name}' must define TOOLS, a list of tools")
    if not all(isinstance(tool, BaseTool) for tool in tools):
        raise ScenarioError("tools.module: every entry in TOOLS must be a tool (see langchain_core.tools.tool)")
    configure = getattr(module, "configure", None)
    if configure is not None:
        try:
            configure(dict(scenario.tool_settings), scenario_dir)
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise ScenarioError(f"tools.module: configure() failed: {type(error).__name__}: {error}") from error
    return {tool.name: tool for tool in tools}


def resolve_tools(scenario: Scenario, registry: dict | None = None) -> dict[str, list]:
    """Route name -> tool objects. A name the module does not provide is a scenario error."""
    registry = load_tools(scenario) if registry is None else registry
    known = f"registered: {', '.join(sorted(registry))}" if registry else "no [tools] module is set"
    resolved = {}
    for name, route in scenario.routes.items():
        for tool_name in route.tools:
            if tool_name not in registry:
                raise ScenarioError(f"routes.{name}.tools: unknown tool '{tool_name}' ({known})")
        resolved[name] = [registry[tool_name] for tool_name in route.tools]
    for tool_name in scenario.confirm_tools:
        if tool_name not in registry:
            raise ScenarioError(f"tools.confirm: unknown tool '{tool_name}' ({known})")
    return resolved


def load_catalogue(scenario: Scenario, registry: dict) -> list[dict] | None:
    """The catalogue tool's entries, called once and checked against the contract; None without [catalogue]."""
    if scenario.catalogue is None:
        return None
    name = scenario.catalogue.tool
    if name not in registry:
        raise ScenarioError(f"catalogue.tool: unknown tool '{name}'")
    where = f"catalogue.tool '{name}'"
    # An adopter's tool can fail in any way; each failure should be one line, not a traceback
    try:
        entries = registry[name].invoke({})
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise ScenarioError(f"{where}: {error}") from error
    if not isinstance(entries, list):
        raise ScenarioError(f"{where} must return a list of entries")
    if not 2 <= len(entries) <= MAX_ENTRIES:
        noun = "entry" if len(entries) == 1 else "entries"
        raise ScenarioError(f"{where} returned {len(entries)} {noun}; it needs from 2 to {MAX_ENTRIES}")
    seen: set = set()
    for number, entry in enumerate(entries, 1):
        _check_entry(entry, f"{where}: entry {number}", seen)
    return entries


def _check_entry(entry, where: str, seen: set) -> None:
    if not isinstance(entry, dict):
        raise ScenarioError(f"{where} must be a table of name, description and facts")
    for key in ("name", "description"):
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            raise ScenarioError(f"{where}: '{key}' must be non-empty text")
    facts = entry.get("facts", [])
    if not isinstance(facts, list) or not all(isinstance(fact, str) for fact in facts):
        raise ScenarioError(f"{where}: 'facts' must be a list of sentences")
    if entry["name"] in seen:
        raise ScenarioError(f"{where}: the name '{entry['name']}' is used twice")
    seen.add(entry["name"])
