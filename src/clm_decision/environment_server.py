"""An environment's named queries, decisions, calculations and scenarios, each offered as its own MCP tool.

Each tool's arguments are the parameters the owner declared, so an agent sees exactly what it can ask for and the
MCP SDK refuses anything else. Every tool only reads: nothing here changes the world. The environment's data is also
offered as resources, read by the same code as the read tools, for clients and people that browse or attach data;
the reads stay tools, since most clients let a model fetch only through tools. server.py serves this when its file is
an environment.
"""
from __future__ import annotations

import contextlib
import inspect
from urllib.parse import quote
from typing import Any, Callable

import pydantic_core
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.server.mcpserver.resources import FunctionResource
from mcp.types import ToolAnnotations

from . import derived
from . import engine
from . import instructions
from . import oauth
from . import tracing
from .dmn import DmnError
from .environment import Environment
from .scenario import ScenarioError

READ_ONLY = ToolAnnotations(read_only_hint=True)
JSON = "application/json"
LISTED_DOCUMENTS = 100      # a collection with at most this many documents lists each as a resource
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


def _tool(run: Callable[[dict], dict], signature: inspect.Signature, trace, name: str) -> Callable[..., dict]:
    """A function with the given signature, which the SDK reads for the tool's schema, calling run(arguments)."""
    def tool(**arguments) -> dict[str, Any]:
        with trace.call(name) if trace else contextlib.nullcontext({}) as record:
            try:
                result = run(arguments)
                record["outcome"] = {"ok": True, **_branch_counts(result)}
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
                                 "limits": list(scenario.limits), "uncertain": list(scenario.uncertain),
                                 "approvals": list(scenario.approvals)}
                          for name, scenario in env.scenarios.items()},
            **({"collections": {name: {"description": collection.description, "filters": collection.filters,
                                       "built": env.store.built(name)}
                                for name, collection in env.collections.items()}} if env.collections else {}),
            **({"topics": env.topics()} if env.topics() else {}),
            **({"derived": _derived(env)} if env.slots else {})}


def _derived(env: Environment) -> dict:
    """Each slot the documents fill, with its status in the derived layer and, when it is not admitted, why."""
    slots = {}
    for name, slot in env.slots.items():
        if slot.kind == "decisions" and name in env.decisions:
            status = env.decisions[name].status
        else:
            status = (env.derived_facts.get(slot.node, {}).get(slot.property) or {}).get("status")
        slots[name] = {"kind": slot.kind, "description": slot.description, "admitted": name in env.admitted,
                       "status": status, **({"why": env.unfilled[name]} if name in env.unfilled else {})}
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


def _add_document_tools(add: Callable, env: Environment, name: str, collection) -> None:
    """search_, document_, cites_ and cited_by_ the collection: they read, and decide nothing."""
    fields = ", ".join(f"{field} ({kind})" for field, kind in collection.filters.items()) or "none"
    text = inspect.Parameter("text", inspect.Parameter.KEYWORD_ONLY, annotation=str)
    ident = inspect.Parameter("id", inspect.Parameter.KEYWORD_ONLY, annotation=str)
    add(f"search_{name}", f"{collection.description} The passages closest to the text, by meaning and by words, "
        f"each with its document id, citation, title, metadata and passage label. Filters, as field: value, a list "
        f"of values, or {{\"from\", \"to\"}} for numbers and dates: {fields}. Passages are evidence to read and "
        "cite by their citation; they decide nothing.",
        lambda args: engine.run_search(env, name, args),
        inspect.Signature([text, _optional("filters", dict[str, Any] | None, None), _optional("limit", int, 10)],
                          return_annotation=_RESULT))
    add(f"document_{name}", f"One document of {name} as stored, with its passages, or only the passage labelled "
        "`passage`: to check what a citation says before relying on it.",
        lambda args: engine.run_document(env, name, args),
        inspect.Signature([ident, _optional("passage", str | None, None)], return_annotation=_RESULT))
    for direction, what in (("cites", "The documents this one cites"),
                            ("cited_by", "The documents that cite this one, with the citing passage")):
        add(f"{direction}_{name}", f"{what}, and theirs to `depth` steps (1 to {engine.MAX_DEPTH}), nearest first.",
            lambda args, direction=direction: engine.run_citations(env, name, args, direction),
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


def in_force(env: Environment) -> dict:
    """The derived entries admitted to decide, each with its origin, status and source; the source's quote only when
    the document tools are offered. The whole layer, with entries not in force, is the portal's."""
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
        cited = {key: value for key, value in (source or {}).items() if env.document_tools or key != "quote"}
        entries[name] = {"kind": slot.kind, "origin": "derived", "status": status, "source": cited, **more}
    return {"admit": list(env.admit), "entries": entries}


def _add_document_resources(server: MCPServer, env: Environment, add: Callable, trace) -> None:
    """Templates for a document and a passage, and every document of a small enough collection listed."""
    def document(collection: str, id: str) -> str:  # pylint: disable=redefined-builtin
        return _json(trace, f"document://{collection}/{id}",
                     lambda: engine.run_document(env, collection, {"id": id}))

    def passage(collection: str, id: str, passage: str) -> str:  # pylint: disable=redefined-builtin
        return _json(trace, f"document://{collection}/{id}/{passage}",
                     lambda: engine.run_document(env, collection, {"id": id, "passage": passage}))
    server.resource("document://{collection}/{id}", name="document", mime_type=JSON,
                    description="One document of a collection with its passages, as document_COLLECTION returns it. "
                                "Each document is also listed when its collection holds at most "
                                f"{LISTED_DOCUMENTS} documents when the server starts.")(document)
    server.resource("document://{collection}/{id}/{passage}", name="passage", mime_type=JSON,
                    description="One passage of a document, by its label, as document_COLLECTION returns it.")(passage)
    for name in env.collections:
        ids = env.store.ids(name, LISTED_DOCUMENTS + 1)
        for ident in ids if len(ids) <= LISTED_DOCUMENTS else ():
            found = env.store.document(name, ident, None)
            add(f"document://{name}/{quote(ident, safe='')}", found["title"], found["citation"],
                lambda name=name, ident=ident: engine.run_document(env, name, {"id": ident}))


def _add_resources(server: MCPServer, env: Environment, trace) -> None:
    """The environment's data as resources: describe, every rule when there is a rule tool, the documents when there
    are document tools, and the derived entries in force when there is a derived layer."""
    def add(uri: str, name: str, description: str, run: Callable[[], dict]) -> None:
        server.add_resource(FunctionResource(uri=uri, name=name, description=description, mime_type=JSON,
                                             fn=lambda: _json(trace, uri, run)))

    add("environment://describe", "describe", "What describe returns: the environment's queries, decisions, "
        "calculations, scenarios, collections, topics and derived slots.", lambda: describe(env))
    if env.topics():
        for name in sorted(name for name, decision in env.decisions.items() if decision.rule):
            add(f"rule://{name}", f"rule {name}",
                f"What the rule tool returns for {name}: {env.decisions[name].description}",
                lambda name=name: explain_rule(env, name))
        server.resource("rule://{name}", name="rule", mime_type=JSON,
                        description="One rule, supplied or derived, with its origin and status, as the rule tool "
                                    "returns it.")(lambda name: _json(trace, f"rule://{name}",
                                                                      lambda: explain_rule(env, name)))
    if env.collections and env.document_tools:
        _add_document_resources(server, env, add, trace)
    if env.slots:
        add("derived://in-force", "derived entries in force", "The derived entries in force: each rule and fact "
            "read from the documents and admitted to decide, with its origin, status and source.",
            lambda: in_force(env))


def _add_prompt(server: MCPServer, env: Environment) -> None:
    """answer_question: the reference agent's instructions with this environment and a question."""
    @server.prompt(name="answer_question", description="Ask a question the way the reference agent does: its "
                   "instructions, this environment's name and description, and the question, in one message.")
    def answer_question(question: str) -> str:
        return instructions.answer_question(env.name, env.description, question)


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
    for name, collection in env.collections.items() if env.document_tools else ():
        _add_document_tools(add, env, name, collection)
    branches = inspect.Parameter("branches", inspect.Parameter.KEYWORD_ONLY, default=None,
                                 annotation=list[dict[str, Any]] | None)
    for name, scenario in env.scenarios.items():
        def run(args: dict, name=name) -> dict:
            chosen = args.pop("branches", None)
            return engine.run_scenario(env, name, args, chosen)
        lever_help = "; ".join(f"{lever}: one of {list(values)}" for lever, values in scenario.levers.items())
        add(f"scenario_{name}", f"{scenario.description} Optional branches, each a value for every lever "
            f"({lever_help}); without them every combination runs.", run, _signature(scenario.params, (branches,)))
    if env.topics():
        add("rules", "The rules that govern a kind of decision (a topic from describe), dependencies first: what each "
            "says, why it exists, who owns it and its version; the information they need, and in to_ask what the "
            "person asking must supply.", lambda args: rules_for(env, args["topic"]), _signature({"topic": "string"}))
        add("rule", "One rule: what it says, why it exists, its owner, version, dependencies, the information it "
            "needs, its parameters and, for a decision table, its rows.",
            lambda args: explain_rule(env, args["name"]), _signature({"name": "string"}))
    add("describe", "The environment: its queries, decisions, calculations and scenarios, with their parameters.",
        lambda _: describe(env), _signature({}))
    _add_resources(server, env, trace)
    _add_prompt(server, env)
    return server
