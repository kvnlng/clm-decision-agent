"""Load a scenario's tools from the Python module it names."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

from .scenario import Scenario, ScenarioError


def _module(scenario_dir: Path, name: str) -> ModuleType:
    """The file beside the scenario, so no other module of that name can stand in for it; without one, the module
    imported by name. A copy already imported from that same file is reused, so its state is the one tests and
    helpers see; any other copy is loaded afresh, as an environment's functions are."""
    path = scenario_dir / f"{name}.py"
    if not path.is_file():
        return importlib.import_module(name)
    imported = sys.modules.get(name)
    found = getattr(imported, "__file__", None)
    if found is not None and Path(found).resolve() == path.resolve():
        return imported
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # In sys.modules while it runs, as an import would be, so code that finds its own module by name (a dataclass,
    # pickling) works; it stays only when no other module holds the name, so a standard one is never replaced
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        _restore(name, imported)
        raise
    if imported is not None:
        _restore(name, imported)
    return module


def _restore(name: str, imported: ModuleType | None) -> None:
    """Put back the module that held the name before, or free the name when none did."""
    if imported is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = imported


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
        module = _module(scenario_dir, name)
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
