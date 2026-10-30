"""Load a scenario's tools from the Python module it names."""
from __future__ import annotations

import importlib
import inspect
import sys
from typing import Callable

from .scenario import Scenario, ScenarioError


def load_tools(scenario: Scenario) -> dict[str, Callable]:
    """Tool name -> function, from the module named in [tools]. The module's configure() is called if it has one."""
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
        raise ScenarioError(f"tools.module: '{name}' must define TOOLS, a list of functions")
    if not all(inspect.isfunction(tool) for tool in tools):
        raise ScenarioError("tools.module: every entry in TOOLS must be a function that takes the request")
    configure = getattr(module, "configure", None)
    if configure is not None:
        try:
            configure(dict(scenario.tool_settings), scenario_dir)
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise ScenarioError(f"tools.module: configure() failed: {type(error).__name__}: {error}") from error
    return {tool.__name__: tool for tool in tools}


def check_tools(scenario: Scenario, registry: dict[str, Callable]) -> None:
    """Every tool the scenario names must be in the registry; otherwise a scenario error naming the setting."""
    known = f"registered: {', '.join(sorted(registry))}" if registry else "no [tools] module is set"
    named = [(f"routes.{route.name}.inform", tool) for route in scenario.routes.values() for tool in route.inform]
    named += [(f"tools.inform.{tool}", tool) for tool in scenario.informs]
    named += [("routing.inform", tool) for tool in scenario.route_inform]
    named += [(f"actions[{index}].tool", action.tool) for index, action in enumerate(scenario.actions)]
    for where, tool in named:
        if tool not in registry:
            raise ScenarioError(f"{where}: unknown tool '{tool}' ({known})")
