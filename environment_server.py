"""An environment's named queries, decisions, calculations and scenarios, each offered as its own MCP tool.

Each tool's arguments are the parameters the owner declared, so an agent sees exactly what it can ask for and the
MCP SDK refuses anything else. Every tool only reads: nothing here changes the world. server.py serves this when
its file is an environment.
"""
from __future__ import annotations

import contextlib
import inspect
from typing import Any, Callable

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

import engine
import oauth
import tracing
from dmn import DmnError
from environment import Environment
from scenario import ScenarioError

READ_ONLY = ToolAnnotations(read_only_hint=True)
_ANNOTATIONS = {"string": str, "number": float, "integer": int, "boolean": bool, "list": list[Any], "any": Any}
_RESULT = dict[str, Any]


def _signature(params: dict[str, str], extra: tuple[inspect.Parameter, ...] = ()) -> inspect.Signature:
    declared = [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=_ANNOTATIONS[kind])
                for name, kind in params.items()]
    return inspect.Signature(declared + list(extra), return_annotation=_RESULT)


def _tool(run: Callable[[dict], dict], signature: inspect.Signature, trace, name: str) -> Callable[..., dict]:
    """A function with the given signature, which the SDK reads for the tool's schema, calling run(arguments)."""
    def tool(**arguments) -> dict[str, Any]:
        with trace.call(name) if trace else contextlib.nullcontext({}) as record:
            try:
                result = run(arguments)
                record["outcome"] = {"ok": True}
                return result
            except (engine.StepError, DmnError, ScenarioError) as error:
                raise ToolError(str(error)) from error
            # The owner's code can fail in any way; the agent should still learn why
            except Exception as error:  # pylint: disable=broad-exception-caught
                raise ToolError(f"{type(error).__name__}: {error}") from error
    tool.__signature__ = signature
    tool.__annotations__ = {**{p.name: p.annotation for p in signature.parameters.values()}, "return": _RESULT}
    return tool


def describe(env: Environment) -> dict:
    """What the environment offers, for an agent to read first."""
    return {"name": env.name, "description": env.description,
            "queries": {name: {"description": query.description, "params": query.params}
                        for name, query in env.queries.items()},
            "decisions": {name: {"description": decision.description, "params": decision.params}
                          for name, decision in env.decisions.items()},
            "calculations": {name: {"description": calculation.description, "params": calculation.params}
                             for name, calculation in env.calculations.items()},
            "judges": {name: {"description": judge.description, "instructions": judge.instructions}
                       for name, judge in env.judges.items()},
            "scenarios": {name: {"description": scenario.description, "params": scenario.params,
                                 "levers": {lever: list(values) for lever, values in scenario.levers.items()},
                                 "objective": scenario.objective, "maximise": scenario.maximise,
                                 "limits": list(scenario.limits)}
                          for name, scenario in env.scenarios.items()}}


def build_environment_server(env: Environment, trace: tracing.TraceWriter | None = None,
                             auth: oauth.OAuth | None = None) -> MCPServer:
    """The MCP server for one environment. With `trace`, each call is written to the trace; with `auth`, HTTP
    callers need a token."""
    server = MCPServer("decision-server", instructions=(
        f"Decisions for '{env.name}'. Call describe first. Every figure and verdict comes from these tools: run a "
        "scenario to compare alternatives rather than combining queries and decisions yourself."),
        log_level="WARNING", **oauth.server_options(auth))

    def add(name: str, description: str, run: Callable[[dict], dict], signature: inspect.Signature) -> None:
        server.add_tool(_tool(run, signature, trace, name), name=name, description=description,
                        annotations=READ_ONLY)

    for name, query in env.queries.items():
        add(f"query_{name}", query.description, lambda args, name=name: engine.run_query(env, name, args),
            _signature(query.params))
    for name, decision in env.decisions.items():
        add(f"decision_{name}", decision.description, lambda args, name=name: engine.run_decision(env, name, args),
            _signature(decision.params))
    for name, calculation in env.calculations.items():
        add(f"calculation_{name}", calculation.description,
            lambda args, name=name: engine.run_calculation(env, name, args), _signature(calculation.params))
    for name, judge in env.judges.items():
        add(f"judge_{name}", f"{judge.description} CLM's probability, 0 to 1, that the answer to "
            f"\"{judge.instructions}\" is yes for the text.",
            lambda args, name=name: engine.run_judgement(env, name, args), _signature({"text": "string"}))
    branches = inspect.Parameter("branches", inspect.Parameter.KEYWORD_ONLY, default=None,
                                 annotation=list[dict[str, Any]] | None)
    for name, scenario in env.scenarios.items():
        def run(args: dict, name=name) -> dict:
            chosen = args.pop("branches", None)
            return engine.run_scenario(env, name, args, chosen)
        lever_help = "; ".join(f"{lever}: one of {list(values)}" for lever, values in scenario.levers.items())
        add(f"scenario_{name}", f"{scenario.description} Optional branches, each a value for every lever "
            f"({lever_help}); without them every combination runs.", run, _signature(scenario.params, (branches,)))
    add("describe", "The environment: its queries, decisions, calculations and scenarios, with their parameters.",
        lambda _: describe(env), _signature({}))
    return server
