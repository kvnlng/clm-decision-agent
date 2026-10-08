"""Load a scenario's tools from the Python module it names."""
from __future__ import annotations

import importlib
import sys

from langchain_core.tools import BaseTool

from scenario import Scenario, ScenarioError


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
