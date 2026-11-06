"""An environment's named queries, decisions, calculations and scenarios, each offered as its own MCP tool.

Each tool's arguments are the parameters the owner declared, so an agent sees exactly what it can ask for and the
MCP SDK refuses anything else. Every tool only reads: nothing here changes the world. The environment's data is also
offered as resources, read by the same code as the read tools, for clients and people that browse or attach data;
the reads stay tools, since most clients let a model fetch only through tools. server.py serves this when its file is
an environment.

The server holds its environment in a Holder (holder.py): each tool call, resource read and prompt takes the current
environment when it starts and keeps it to its end, so `reload` can put a new one in place while calls run.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import pydantic_core
from mcp.server import MCPServer
from mcp.server.caching import CacheHint
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.server.mcpserver.prompts import Prompt
from mcp.server.mcpserver.resources import FunctionResource
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS, ToolAnnotations

from . import admin as admin_module
from . import derived
from . import engine
from . import instructions
from . import oauth
from . import serving
from . import tracing
from .dmn import DmnError
from .environment import NOT_DERIVED, Environment
from .holder import Holder
from .scenario import ScenarioError
from .serving import Serve

READ_ONLY = ToolAnnotations(read_only_hint=True)
JSON = "application/json"
_ANNOTATIONS = {"string": str, "number": float, "integer": int, "boolean": bool, "list": list[Any], "any": Any}
_RESULT = dict[str, Any]


def _signature(params: dict[str, str], extra: tuple[inspect.Parameter, ...] = ()) -> inspect.Signature:
    declared = [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=_ANNOTATIONS[kind])
                for name, kind in params.items()]
    return inspect.Signature(declared + list(extra), return_annotation=_RESULT)


def _branch_counts(result: dict) -> dict:
    """For a scenario's result, how many branches were kept, rejected and failed, and the best; else nothing."""
    if "branches" not in result:
        return {}
    statuses = [branch["status"] for branch in result["branches"]]
    return {**{status: statuses.count(status) for status in ("kept", "rejected", "failed")}, "best": result["best"]}


@dataclass
class _Offering:
    """What a server built here serves, kept on it as `server.offering` so reload can choose again: the holder, how
    tools are chosen, the trace, the operator's disabled tools, the tools offered now and the resources and templates
    added, which stay."""
    holder: Holder
    serve: Serve
    trace: tracing.TraceWriter | None
    admin: admin_module.Admin | None = None
    offered: list[str] = field(default_factory=list)
    resources: set[str] = field(default_factory=set)
    reloading: threading.Lock = field(default_factory=threading.Lock)     # one reload at a time; calls never wait


def _tool(run: Callable[[Environment, dict], dict], signature: inspect.Signature, offering: _Offering,
          name: str) -> Callable[..., dict]:
    """A function with the given signature, which the SDK reads for the tool's schema, calling run(env, arguments)
    with the environment current when the call starts, held to its end."""
    trace = offering.trace

    def tool(**arguments) -> dict[str, Any]:
        # The arguments are the request's text: written only with --trace-text
        text = json.dumps(arguments, default=str) if trace and trace.text else None
        with trace.call(name, text) if trace else contextlib.nullcontext({}) as record:
            try:
                with offering.holder.use() as env:
                    result = run(env, arguments)
                record["outcome"] = {"ok": True, **_branch_counts(result)}
                if trace:
                    record["result_size_bytes"] = len(json.dumps(result, default=str).encode())
                return result
            except (engine.StepError, DmnError, ScenarioError) as error:
                raise ToolError(str(error)) from error
            # The owner's code can fail in any way; the agent should still learn why
            except Exception as error:  # pylint: disable=broad-exception-caught
                raise ToolError(f"{type(error).__name__}: {error}") from error
    tool.__signature__ = signature
    tool.__annotations__ = {**{p.name: p.annotation for p in signature.parameters.values()}, "return": _RESULT}
    return tool


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def describe(env: Environment, offered: list[str] | None = None, left_out: list[str] | None = None,
             read_only: bool = False, disabled: list[str] | None = None) -> dict:
    """What the environment offers, for an agent to read first. With `offered`, only the parts whose tools are
    offered, in left_out the tools the operator left out and in disabled those offered but disabled, whose parts are
    not shown; read_only says the server is read-only."""
    def shown(tool: str) -> bool:
        return (offered is None or tool in offered) and tool not in (disabled or ())
    collections = {name: collection for name, collection in env.collections.items() if env.document_tools
                   and any(shown(f"{kind}_{name}") for kind in ("search", "document", "cites", "cited_by"))}
    return {"name": env.name, "description": env.description,
            "queries": {name: {"description": query.description, "params": query.params}
                        for name, query in env.queries.items() if shown(f"query_{name}")},
            "decisions": {name: {"description": decision.description, "params": decision.params}
                          for name, decision in env.decisions.items() if shown(f"decision_{name}")},
            "calculations": {name: {"description": calculation.description, "params": calculation.params}
                             for name, calculation in env.calculations.items() if shown(f"calculation_{name}")},
            "judges": {name: {"description": judge.description, "instructions": judge.instructions}
                       for name, judge in env.judges.items() if shown(f"judge_{name}")},
            "scenarios": {name: {"description": scenario.description, "params": scenario.params,
                                 "levers": {lever: list(values) for lever, values in scenario.levers.items()},
                                 "objective": scenario.objective, "maximise": scenario.maximise,
                                 "limits": list(scenario.limits), "uncertain": list(scenario.uncertain),
                                 "approvals": list(scenario.approvals)}
                          for name, scenario in env.scenarios.items() if shown(f"scenario_{name}")},
            **({"collections": {name: {"description": collection.description, "filters": collection.filters,
                                       "built": env.store.built(name)}
                                for name, collection in collections.items()}} if collections else {}),
            **({"topics": env.topics()} if env.topics() and shown("rules") else {}),
            **({"derived": _derived(env)} if env.slots else {}),
            **({"left_out": left_out} if left_out else {}), **({"disabled": disabled} if disabled else {}),
            **({"read_only": True} if read_only else {})}


def _derived(env: Environment) -> dict:
    """Each slot the documents fill, with its status in the derived layer (not derived when it has no entry) and,
    when it is not admitted, why."""
    slots = {name: {"kind": slot.kind, "description": slot.description, "admitted": name in env.admitted,
                    "status": env.slot_statuses.get(name, NOT_DERIVED),
                    **({"why": env.unfilled[name]} if name in env.unfilled else {})}
             for name, slot in env.slots.items()}
    note = (f"Derived entries are admitted with status {', '.join(env.admit)}, for testing only: some are not "
            "approved by a person." if env.admit_override else
            f"Derived entries with status {', '.join(env.admit)} are admitted: the checked ones are not approved by a "
            "person." if set(env.admit) - {"approved"} else
            f"Only derived entries with status {', '.join(env.admit)} are admitted.")
    reader = env.second_reader
    second = (f"clm-derive check makes a second reading by {reader['model']}: it derives each entry again from its "
              "document, without the first answer, and an entry it disagrees with fails the check. Agreement is not "
              "proof: two readings can make the same mistake." if reader else
              "There is no second reading ([derived] names no second_reader): each entry passed only the mechanical "
              "check, which cannot tell whether a figure is in the right place.")
    return {"admit": list(env.admit), "note": note, "second_reader": reader, "second_reading": second,
            "slots": slots}


def _information(env: Environment, name: str) -> dict:
    item = env.information[name]
    return {"name": name, "description": item.description, "kind": item.kind, "query": item.query,
            "unit": item.unit}


def rules_for(env: Environment, topic: str) -> dict:
    """The rules governing a kind of decision, and the rules they depend on, dependencies first; the information they
    need, and which of it the person asking must supply."""
    topics = env.topics()
    if topic not in topics:
        raise engine.StepError(f"no rules govern '{topic}'; the topics are: {', '.join(topics)}")
    ordered: list[str] = []

    def visit(name: str) -> None:
        if name not in ordered:
            for other in env.decisions[name].rule.depends_on:
                visit(other)
            ordered.append(name)
    for name in sorted(env.decisions):
        rule = env.decisions[name].rule
        if rule and topic in rule.governs:
            visit(name)
    needed = list(dict.fromkeys(item for name in ordered for item in env.decisions[name].rule.needs))
    information = [_information(env, item) for item in needed]
    return {"topic": topic, "rules": [engine.rule_info(env, name) for name in ordered], "information": information,
            "to_ask": [item for item in information if item["kind"] == "supplied"]}


def explain_rule(env: Environment, name: str) -> dict:
    """One rule, with its parameters and, for a DMN table, its rows as written."""
    if name not in env.decisions or env.decisions[name].rule is None:
        known = ", ".join(sorted(n for n, d in env.decisions.items() if d.rule))
        raise engine.StepError(f"no rule '{name}'; the rules are: {known}")
    decision = env.decisions[name]
    result = {**engine.rule_info(env, name), "params": decision.params}
    if decision.table is not None:
        table = decision.table
        result["rows"] = [{"id": row.id, "when": dict(zip(table.inputs, row.entries)),
                           "then": dict(zip(table.outputs, row.values))} for row in table.rules]
    return result


def _optional(name: str, annotation, default) -> inspect.Parameter:
    return inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation)


def _add_document_tools(add: Callable, name: str, collection) -> None:
    """search_, document_, cites_ and cited_by_ the collection: they read, and decide nothing."""
    fields = ", ".join(f"{field} ({kind})" for field, kind in collection.filters.items()) or "none"
    text = inspect.Parameter("text", inspect.Parameter.KEYWORD_ONLY, annotation=str)
    ident = inspect.Parameter("id", inspect.Parameter.KEYWORD_ONLY, annotation=str)
    add(f"search_{name}", f"{collection.description} The passages closest to the text, by meaning and by words, "
        f"each with its document id, citation, title, metadata and passage label. Filters, as field: value, a list "
        f"of values, or {{\"from\", \"to\"}} for numbers and dates: {fields}. Passages are evidence to read and "
        "cite by their citation; they decide nothing.",
        lambda env, args: engine.run_search(env, name, args),
        inspect.Signature([text, _optional("filters", dict[str, Any] | None, None), _optional("limit", int, 10)],
                          return_annotation=_RESULT))
    add(f"document_{name}", f"One document of {name} as stored, with its passages, or only the passage labelled "
        "`passage`: to check what a citation says before relying on it.",
        lambda env, args: engine.run_document(env, name, args),
        inspect.Signature([ident, _optional("passage", str | None, None)], return_annotation=_RESULT))
    for direction, what in (("cites", "The documents this one cites, each with citing_passage, the passage citing it, "
                                      "in citing_document"),
                            ("cited_by", "The documents that cite this one, with the citing passage")):
        add(f"{direction}_{name}", f"{what}, and theirs to `depth` steps (1 to {engine.MAX_DEPTH}), nearest first.",
            lambda env, args, direction=direction: engine.run_citations(env, name, args, direction),
            inspect.Signature([ident, _optional("depth", int, 1)], return_annotation=_RESULT))


def _json(trace, uri: str, run: Callable[[], dict]) -> str:
    """A resource's content: what run returns, as JSON the way a tool's result is; a failure is a ResourceError
    carrying the one-line reason, as a tool's is a ToolError. With `trace`, the read is written to it like a call,
    named by its URI."""
    with trace.call(uri) if trace else contextlib.nullcontext({}) as record:
        try:
            found = pydantic_core.to_json(run(), fallback=str).decode()
            record["outcome"] = {"ok": True}
            return found
        except (engine.StepError, derived.DerivedError) as error:
            raise ResourceError(str(error)) from error
        # The owner's code and the store can fail in any way; the client should still learn why
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise ResourceError(f"{type(error).__name__}: {error}") from error


def in_force(env: Environment, readable: list[str] | None = None) -> dict:
    """The derived entries admitted to decide, each with its origin, status and source; the source's quote only when
    the document tools are offered, and with `readable`, only when the entry's collection is among them. The whole
    layer, with entries not in force, is the portal's."""
    entries = {}
    for name in env.admitted:
        slot = env.slots[name]
        if slot.kind == "decisions":
            decision = env.decisions[name]
            status, source, more = decision.status, decision.source, {}
        else:
            entry = env.derived_facts[slot.node][slot.property]
            status, source = entry.get("status"), entry.get("source")
            more = {"node": slot.node, "property": slot.property, "value": entry.get("value")}
        quotes = env.document_tools if readable is None else slot.collection in readable
        cited = {key: value for key, value in (source or {}).items() if quotes or key != "quote"}
        entries[name] = {"kind": slot.kind, "origin": "derived", "status": status, "source": cited, **more}
    return {"admit": list(env.admit), "entries": entries}


def _found(env: Environment, offering: _Offering) -> list[tuple[str, str, Callable, inspect.Signature]]:
    """Every tool the environment can offer, as (name, description, run, signature)."""
    found: list[tuple[str, str, Callable, inspect.Signature]] = []
    _all_tools(env, lambda *tool: found.append(tool), lambda current: _described(offering, current))
    return found


def _chosen(env: Environment, offering: _Offering) -> tuple[list[str], list[str]]:
    """The tools offered for `env` and those left out, chosen as the server chooses them."""
    return serving.choose(offering.serve, [name for name, *_ in _found(env, offering)])


def _described(offering: _Offering, env: Environment) -> dict:
    offered, left_out = _chosen(env, offering)
    held = offering.admin.disabled if offering.admin else frozenset()
    return describe(env, offered, left_out, offering.serve.read_only, [name for name in offered if name in held])


def _readable(env: Environment, offering: _Offering) -> list[str]:
    """The collections whose document tool is offered for `env`."""
    offered = _chosen(env, offering)[0]
    return [name for name in env.collections if f"document_{name}" in offered]


def _read(offering: _Offering, uri: str, run: Callable[[Environment], dict]) -> str:
    """A resource read: run with the environment current when the read starts, as JSON."""
    with offering.holder.use() as env:
        return _json(offering.trace, uri, lambda: run(env))


def _rule(offering: _Offering, env: Environment, name: str) -> dict:
    if "rule" not in _chosen(env, offering)[0]:
        raise engine.StepError("this server offers no rules")
    return explain_rule(env, name)


def _document(offering: _Offering, env: Environment, collection: str, args: dict) -> dict:
    readable = _readable(env, offering)
    if collection not in readable:
        raise engine.StepError(f"no collection '{collection}' (the environment has: {', '.join(sorted(readable))})")
    return engine.run_document(env, collection, args)


def _add_resources(server: MCPServer, offering: _Offering, env: Environment, offered: list[str]) -> None:
    """The environment's data as resources, each read with the environment current at the time: describe; the rule
    template when the rule tool is offered; the document templates when a collection's document tool is; and the
    derived entries in force when the environment has a [derived] table. The SDK cannot remove a resource, so only
    those not yet added are added, and one a reload no longer offers says so when read."""
    def missing(uri: str) -> bool:
        if uri in offering.resources:
            return False
        offering.resources.add(uri)
        return True

    def fixed(uri: str, name: str, description: str, run: Callable[[Environment], dict]) -> None:
        if missing(uri):
            server.add_resource(FunctionResource(uri=uri, name=name, description=description, mime_type=JSON,
                                                 fn=lambda: _read(offering, uri, run)))

    def rule(name: str) -> str:
        return _read(offering, f"rule://{name}", lambda current: _rule(offering, current, name))

    def document(collection: str, id: str) -> str:  # pylint: disable=redefined-builtin
        return _read(offering, f"document://{collection}/{id}",
                     lambda current: _document(offering, current, collection, {"id": id}))

    def passage(collection: str, id: str, passage: str) -> str:  # pylint: disable=redefined-builtin
        return _read(offering, f"document://{collection}/{id}/{passage}",
                     lambda current: _document(offering, current, collection, {"id": id, "passage": passage}))

    fixed("environment://describe", "describe", "What describe returns: the environment's queries, decisions, "
          "calculations, scenarios, collections, topics and derived slots.",
          lambda current: _described(offering, current))
    if "rule" in offered and missing("rule://{name}"):
        server.resource("rule://{name}", name="rule", mime_type=JSON,
                        description="One rule, supplied or derived, with its origin and status, as the rule tool "
                                    "returns it.")(rule)
    if any(f"document_{name}" in offered for name in env.collections) and missing("document://{collection}/{id}"):
        server.resource("document://{collection}/{id}", name="document", mime_type=JSON,
                        description="One document of a collection with its passages, as document_COLLECTION returns "
                                    "it.")(document)
        server.resource("document://{collection}/{id}/{passage}", name="passage", mime_type=JSON,
                        description="One passage of a document, by its label, as document_COLLECTION returns it."
                        )(passage)
    if env.slots or env.admit:
        fixed("derived://in-force", "derived entries in force", "The derived entries in force: each rule and fact "
              "read from the documents and admitted to decide, with its origin, status and source.",
              lambda current: in_force(current, _readable(current, offering)))


class _QuestionPrompt(Prompt):
    """A prompt whose question is required and not blank: refused with one error line before the SDK renders it,
    which would log a traceback for a missing argument and accept an empty one."""

    async def render(self, arguments, context):
        question = (arguments or {}).get("question")
        if not isinstance(question, str) or not question.strip():
            raise MCPError(INVALID_PARAMS, f"{self.name} needs a question: give it the question to ask")
        return await super().render(arguments, context)


def _add_prompt(server: MCPServer, offering: _Offering) -> None:
    """answer_question: the reference agent's instructions with the current environment and a question."""
    def answer_question(question: str) -> str:
        with offering.holder.use() as env:
            return instructions.answer_question(env.name, env.description, question, _chosen(env, offering)[0])
    server.add_prompt(_QuestionPrompt.from_function(
        answer_question, name="answer_question", description="Ask a question the way the reference agent does: its "
        "instructions, this environment's name and description, and the question, in one message."))


def _instructions(env: Environment) -> str:
    """The server's instructions. They are fixed when the server is built, so they name no tool a reload can take
    away: describe is always offered, and a scenario is named only through what describe lists."""
    return (f"Decisions for '{env.name}'. Call describe first. Every figure and verdict comes from these tools: when "
            "describe lists a scenario, run it to compare alternatives rather than combining queries and decisions "
            "yourself.")


def _offer_tools(server: MCPServer, offering: _Offering, found: list, offered: list[str]) -> None:
    """Registers the tools named in `offered`, each replacing one of the same name, since add_tool would keep the
    old one, and removes the tools no longer offered."""
    for name in offering.offered:
        if name not in offered:
            server.remove_tool(name)
    for name, description, run, signature in found:
        if name in offered:
            if name in offering.offered:
                server.remove_tool(name)
            server.add_tool(_tool(run, signature, offering, name), name=name, description=description,
                            annotations=READ_ONLY)
    offering.offered = list(offered)


# pylint: disable-next=too-many-arguments
def build_environment_server(env_or_holder: Environment | Holder, trace: tracing.TraceWriter | None = None,
                             auth: oauth.OAuth | None = None, serve: Serve = Serve(),
                             admin: admin_module.Admin | None = None, **server_options) -> MCPServer:
    """The MCP server for an environment, or for the one a Holder holds, which `reload` can replace while it runs;
    server.holder is the holder. With `trace`, each call is written to the trace; with `auth`, HTTP callers need a
    token; `server_options` go to MCPServer. `serve` chooses the tools offered (serving.py; an unknown one is a
    ScenarioError); every tool here only reads, so read-only changes nothing but describe. A tool that writes, if
    one is ever added, must not run when serve.read_only is set. With `admin`, the tools it disables are hidden and
    refused (admin.py), tools/list carries a 30-second ttlMs and a change to the list is announced on admin.bus."""
    holder = env_or_holder if isinstance(env_or_holder, Holder) else Holder(env_or_holder)
    env = holder.current
    offering = _Offering(holder, serve, trace, admin)
    found = _found(env, offering)
    offered, _ = serving.choose(serve, [name for name, *_ in found])
    if admin:
        admin.known = lambda: set(offering.offered)
        server_options = {"middleware": [admin_module.middleware(admin)], "subscriptions": admin.bus,
                          "cache_hints": {"tools/list": CacheHint(ttl_ms=30_000)}, **server_options}
    server = MCPServer("decision-server", instructions=_instructions(env), log_level="WARNING",
                       **oauth.server_options(auth), **server_options)
    server.holder = holder
    server.offering = offering
    _offer_tools(server, offering, found, offered)
    _add_resources(server, offering, env, offered)
    _add_prompt(server, offering)
    return server


def reload(server: MCPServer, new: Environment) -> dict:
    """Serves `new` in place of the server's environment: calls already running finish with the old one, which is
    closed when the last ends. The tools are chosen again; each offered one is registered again, so it has the new
    environment's description and parameters, and the result names the tools added and removed. When the choice
    fails for `new` (a tool named in --tools is gone), the old environment keeps serving, `new` is closed and the
    error is raised. When the tools change on a server with an admin, the change is announced on admin.bus from the
    running event loop."""
    offering: _Offering = server.offering
    with offering.reloading:
        try:
            found = _found(new, offering)
            offered, _ = serving.choose(offering.serve, [name for name, *_ in found])
        except BaseException:
            if new is not offering.holder.current:
                new.close(keeping=offering.holder.current)
            raise
        offering.holder.swap(new)
        before = offering.offered
        _offer_tools(server, offering, found, offered)
        _add_resources(server, offering, new, offered)
    if offering.admin and offered != before:
        offering.admin.announce()
    return {"added": [name for name in offered if name not in before],
            "removed": [name for name in before if name not in offered]}


def _all_tools(env: Environment, add: Callable, described: Callable[[Environment], dict]) -> None:
    """Every tool the environment can offer, each given to add(name, description, run, signature); run takes the
    environment current when the call starts, and the arguments, and describe runs `described` on it."""

    for name, query in env.queries.items():
        add(f"query_{name}", query.description, lambda env, args, name=name: engine.run_query(env, name, args),
            _signature(query.params))
    for name, decision in env.decisions.items():
        add(f"decision_{name}", decision.description, lambda env, args, name=name: engine.run_decision(env, name, args),
            _signature(decision.params))
    for name, calculation in env.calculations.items():
        add(f"calculation_{name}", calculation.description,
            lambda env, args, name=name: engine.run_calculation(env, name, args), _signature(calculation.params))
    for name, judge in env.judges.items():
        add(f"judge_{name}", f"{judge.description} CLM's probability, 0 to 1, that the answer to "
            f"\"{judge.instructions}\" is yes for the text.",
            lambda env, args, name=name: engine.run_judgement(env, name, args), _signature({"text": "string"}))
    for name, collection in env.collections.items() if env.document_tools else ():
        _add_document_tools(add, name, collection)
    branches = inspect.Parameter("branches", inspect.Parameter.KEYWORD_ONLY, default=None,
                                 annotation=list[dict[str, Any]] | None)
    for name, scenario in env.scenarios.items():
        def run(env: Environment, args: dict, name=name) -> dict:
            chosen = args.pop("branches", None)
            return engine.run_scenario(env, name, args, chosen)
        lever_help = "; ".join(f"{lever}: one of {list(values)}" for lever, values in scenario.levers.items())
        add(f"scenario_{name}", f"{scenario.description} Optional branches, each a value for every lever "
            f"({lever_help}); without them every combination runs.", run, _signature(scenario.params, (branches,)))
    if env.topics():
        add("rules", "The rules that govern a kind of decision (a topic from describe), dependencies first: what each "
            "says, why it exists, who owns it and its version; the information they need, and in to_ask what the "
            "person asking must supply.", lambda env, args: rules_for(env, args["topic"]),
            _signature({"topic": "string"}))
        add("rule", "One rule: what it says, why it exists, its owner, version, dependencies, the information it "
            "needs, its parameters and, for a decision table, its rows.",
            lambda env, args: explain_rule(env, args["name"]), _signature({"name": "string"}))
    add("describe", "The environment: its queries, decisions, calculations and scenarios, with their parameters.",
        lambda env, args: described(env), _signature({}))
