"""The decision server: a scenario's decisions offered to other agents as MCP tools, over stdio or HTTP."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hmac
import os
import sys
import threading
import time
from typing import Any, Callable

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.responses import JSONResponse

import decision
import oauth
import tracing
from clm_common import ClmError
from evaluate import agreement as replay_agreement
from dataset import labels
from record import load_records, record_key, save_correction
from environment import Environment, is_environment, load_environment
from environment_server import build_environment_server
from scenario import Scenario, ScenarioError, load_scenario
from tools import check_tools, load_tools

KEY_ENV = "DECISION_SERVER_KEY"
MAX_BATCH = 1000
READ_ONLY = ToolAnnotations(read_only_hint=True)
CHANGES = ToolAnnotations(read_only_hint=False)


class _OneAtATime:
    """The router, with each call to CLM taking its turn. Tools run outside the lock, so a slow one holds up no one."""

    def __init__(self, router, lock: threading.Lock):
        self._router, self._lock = router, lock

    def __getattr__(self, name: str):
        method = getattr(self._router, name)

        def locked(*args, **kwargs):
            start = time.perf_counter()
            with self._lock:
                tracing.add("lock_wait_ms", (time.perf_counter() - start) * 1000)
                return method(*args, **kwargs)
        return locked


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _corrected(scenario: Scenario, tools: dict[str, Callable], text: str, route: str, context: dict | None,
               answers: dict | None, option: str | None, earlier: list[dict]) -> dict:
    """The answers, facts, options and chosen option to save with a correction; ToolError if one is wrong."""
    if answers is None and option is None:
        return {}
    if route not in scenario.routes:
        raise ToolError(f"unknown route '{route}'; the routes are {', '.join(scenario.routes)}")
    try:
        labels(scenario, {"answers": answers or {}})
    except ValueError as error:
        raise ToolError(str(error)) from error
    spec = scenario.routes[route].options
    if option is not None and spec is None:
        raise ToolError(f"routes.{route} has no options to choose among")
    # The route's informing tools find the facts and options, as they do when deciding
    request: dict[str, Any] = {"text": text, "route": route, "answers": {}, "rules": [], "facts": []}
    if context is not None:
        request["context"] = context
    informed = decision.gather(request, scenario, tools, earlier)
    more: dict[str, Any] = {"facts": {outcome["tool"]: outcome["facts"] for outcome in informed
                                      if outcome["status"] == "done" and outcome.get("facts")},
                            **({"answers": answers} if answers is not None else {})}
    if option is not None:
        found = next((outcome for outcome in informed if outcome["tool"] == spec.tool), {})
        offered = found.get("options", []) if found.get("status") == "done" else []
        if option not in [candidate["id"] for candidate in offered]:
            raise ToolError(f"option {option!r} was not among the options {spec.tool} found")
        more.update(options=offered, option=option)
    return more


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def build_server(scenario: Scenario, router, tools: dict[str, Callable], review: bool,  # pylint: disable=too-many-arguments
                 trace: tracing.TraceWriter | None = None, auth: oauth.OAuth | None = None) -> MCPServer:
    """The MCP server for one scenario. Review mode adds `correct` and `agreement`, which write and read the record.
    With `trace`, each call is written to the trace; with `auth`, HTTP callers need a token."""
    server = MCPServer("decision-server", instructions=(
        f"Decisions for the scenario '{scenario.path.stem}'. Call describe to see the routes and questions, "
        "then decide with the text of a request. decide may run the scenario's acting tools."),
        log_level="WARNING", **oauth.server_options(auth))
    # CLM runs on one model and calls arrive on worker threads, so its calls take turns
    router = _OneAtATime(router, threading.Lock())

    def clm(work: Callable[[], dict]) -> dict:
        try:
            return work()
        except ClmError as error:
            raise ToolError(str(error)) from error

    def traced(tool: str, text: str | None = None):
        return trace.call(tool, text) if trace else contextlib.nullcontext({})

    @server.tool(annotations=CHANGES)
    def decide(text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Decide on a request: its route and confidence, the answers to the scenario's questions, the rules
        that apply, facts the informing tools found, and what the acting tools did. An uncertain route runs no
        acting tool. `context` (such as a booking or user ID) is given to the scenario's tools, not to CLM."""
        with traced("decide", text) as record:
            result = clm(lambda: decision.decide(text, scenario, router, tools, context))
            if trace:
                record["outcome"] = {key: result.get(key) for key in ("route", "confidence", "uncertain", "source")}
            return result

    @server.tool(annotations=CHANGES)
    def decide_many(requests: list[dict[str, Any]]) -> dict[str, Any]:
        """Decide on many requests at once, each {"text": ..., "context": ...} with context optional: the same
        decisions as decide, in the same order, faster per request. At most 1000 requests a call."""
        if len(requests) > MAX_BATCH:
            raise ToolError(f"decide_many takes at most {MAX_BATCH} requests a call, not {len(requests)}")
        for i, request in enumerate(requests):
            if not isinstance(request.get("text"), str) or not isinstance(request.get("context") or {}, dict):
                raise ToolError(f"requests[{i}] must be {{'text': ..., 'context': {{...}}}}, with context optional")
        with traced("decide_many"):
            return {"decisions": clm(lambda: decision.decide_many(requests, scenario, router, tools))}

    @server.tool(annotations=READ_ONLY)
    def check_call(text: str, call: dict[str, Any]) -> dict[str, Any]:
        """CLM's opinion of a tool call you propose for a request: a probability and whether it is approved."""
        with traced("check_call", text) as record:
            result = clm(lambda: decision.check_call(text, call, scenario, router))
            record["outcome"] = result
            return result

    @server.tool(annotations=READ_ONLY)
    def describe() -> dict[str, Any]:
        """The decisions this server makes: its routes, questions and rules."""
        with traced("describe"):
            return decision.describe(scenario)

    if not review:
        return server

    @server.tool(annotations=CHANGES)
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def correct(text: str, route: str, context: dict[str, Any] | None = None, answers: dict[str, Any] | None = None,
                option: str | None = None) -> dict[str, Any]:
        """Review mode: save the right route for a request, for training. Near-identical requests get it at once.
        Send the same `context` as with decide, so the route is saved with the facts it was decided on.
        Optionally also the right `answers` to the questions (true/false, a level's number, or a choice's key)
        and the right `option`'s id, on a route with options."""
        def work() -> dict:
            earlier = decision.route_facts(text, scenario, tools, context)
            # The route is saved and reused for the text it was decided on: the request and routing's facts
            seen = decision.routed_text(text, decision.facts_of(earlier))
            more = _corrected(scenario, tools, text, route, context, answers, option, earlier)
            made = router.route(seen, scenario)
            try:
                save_correction(scenario, seen, route, {"route": made.get("clm_route", made["route"]),
                                                        "confidence": made["confidence"]}, more)
            except ValueError as error:
                raise ToolError(str(error)) from error
            if scenario.live_corrections:
                router.add_verdict(seen, route, scenario)
            return {"saved": True}
        with traced("correct", text):
            return clm(work)

    @server.tool(annotations=READ_ONLY)
    def agreement() -> dict[str, Any]:
        """Review mode: how often CLM's own route agrees with the saved corrections."""
        records, _ = load_records(scenario.record_path, record_key(scenario))
        # Replayed without reusing corrections, or every correction would agree with itself
        plain = dataclasses.replace(scenario, live_corrections=False)
        with traced("agreement"):
            return clm(lambda: replay_agreement(records, plain, router))

    return server


def http_app(server: MCPServer, key: str | None, host: str = "127.0.0.1", allowed_hosts: list[str] | None = None):
    """The server over Streamable HTTP at /mcp, answering only requests that send `Authorization: Bearer <key>`.
    With no key, the server was built with OAuth and checks tokens itself."""
    security = TransportSecuritySettings(allowed_hosts=allowed_hosts) if allowed_hosts else None
    app = server.streamable_http_app(stateless_http=True, host=host, transport_security=security)
    if key is None:
        return app
    expected = f"Bearer {key}".encode()

    async def guarded(scope, receive, send):
        if scope["type"] == "http":
            sent = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(sent, expected):
                await JSONResponse({"error": "missing or wrong key"}, status_code=401)(scope, receive, send)
                return
        await app(scope, receive, send)
    return guarded


def load(path: str):
    """-> (scenario, router, tools), with CLM loaded and earlier corrections ready to reuse; raises ScenarioError."""
    scenario = load_scenario(path)
    tools = load_tools(scenario)
    check_tools(scenario, tools)
    key = record_key(scenario)
    from clm_router import make_router  # pylint: disable=import-outside-toplevel
    print("Loading CLM...", file=sys.stderr)
    router, note = make_router(scenario)
    print(f"CLM is using {note}.", file=sys.stderr)
    if scenario.live_corrections:
        records, _ = load_records(scenario.record_path, key)
        if router.load_verdicts(records, scenario):
            print(f"Reusing {len(router.verdicts)} corrections from {scenario.record_path.name}.", file=sys.stderr)
    return scenario, router, tools


def load_any(path: str):
    """An environment, loaded without CLM, or a CLM scenario's (scenario, router, tools) as `load` gives them."""
    return load_environment(path) if is_environment(path) else load(path)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code. Messages go to stderr: stdout is MCP's."""
    parser = argparse.ArgumentParser(description="Offer a scenario's or an environment's decisions to other agents "
                                                 "over MCP.")
    parser.add_argument("scenario", help="path to the scenario file, or to an environment's environment.toml")
    parser.add_argument("--review", action="store_true", help="also offer correct and agreement")
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--auth", choices=("key", "oauth"), default="key",
                        help="over HTTP: a shared key (default), or tokens from an OIDC issuer")
    parser.add_argument("--key-env", default=KEY_ENV, help="variable holding the key callers must send")
    parser.add_argument("--issuer", help=f"with --auth oauth: the token issuer's URL (or {oauth.ISSUER_ENV})")
    parser.add_argument("--audience", help=f"with --auth oauth: the audience tokens must name (or "
                                           f"{oauth.AUDIENCE_ENV}; default: the resource URL)")
    parser.add_argument("--scope", action="append", dest="scopes",
                        help=f"with --auth oauth: a scope or app role tokens must carry (repeatable, or "
                             f"{oauth.SCOPES_ENV}, space-separated)")
    parser.add_argument("--resource-url", help="with --auth oauth: this server's public MCP address (default: "
                                               "http://HOST:PORT/mcp)")
    parser.add_argument("--allowed-host", action="append", dest="allowed_hosts",
                        help="a Host header to accept, such as decisions.example.org:443 (repeatable)")
    parser.add_argument("--trace", metavar="FILE", help="write one line per call to FILE, saying where the time went")
    parser.add_argument("--trace-text", action="store_true", help="include each request's text in the trace")
    args = parser.parse_args(argv)
    key, auth = None, None
    if args.http and args.auth == "oauth":
        try:
            auth = oauth.from_args(args.issuer, args.audience, args.scopes,
                                   args.resource_url or f"http://{args.host}:{args.port}/mcp")
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    elif args.http:
        key = os.environ.get(args.key_env, "").strip()
        if not key:
            print(f"error: set {args.key_env} to the key callers must send", file=sys.stderr)
            return 2
    try:
        if args.review and is_environment(args.scenario):
            print("error: review mode is for CLM scenarios; an environment has no corrections to review",
                  file=sys.stderr)
            return 2
        loaded = load_any(args.scenario)
    except (ScenarioError, ClmError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    trace = tracing.TraceWriter(args.trace, text=args.trace_text) if args.trace else None
    if isinstance(loaded, Environment):
        server = build_environment_server(loaded, trace=trace, auth=auth)
    else:
        scenario, router, tools = loaded
        server = build_server(scenario, router, tools, args.review, trace=trace, auth=auth)
    if not args.http:
        server.run("stdio")
        return 0
    import uvicorn  # pylint: disable=import-outside-toplevel
    print(f"Serving MCP at http://{args.host}:{args.port}/mcp"
          + (f", taking tokens from {auth.settings.issuer_url}" if auth else ""), file=sys.stderr)
    uvicorn.run(http_app(server, key, args.host, args.allowed_hosts), host=args.host, port=args.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
