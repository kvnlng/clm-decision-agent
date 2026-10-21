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
import tracing
from clm_common import ClmError
from evaluate import agreement as replay_agreement
from record import load_records, record_key, save_correction
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
def build_server(scenario: Scenario, router, tools: dict[str, Callable], review: bool,
                 trace: tracing.TraceWriter | None = None) -> MCPServer:
    """The MCP server for one scenario. Review mode adds `correct` and `agreement`, which write and read the record.
    With `trace`, each call is written to the trace."""
    server = MCPServer("decision-server", instructions=(
        f"Decisions for the scenario '{scenario.path.stem}'. Call describe to see the routes and questions, "
        "then decide with the text of a request. decide may run the scenario's acting tools."),
        log_level="WARNING")
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
            if not isinstance(request.get("text"), str) or not isinstance(request.get("context", {}), dict):
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
    def correct(text: str, route: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Review mode: save the right route for a request, for training. Near-identical requests get it at once.
        Send the same `context` as with decide, so the route is saved with the facts it was decided on."""
        def work() -> dict:
            # The route is saved and reused for the text it was decided on: the request and routing's facts
            seen = decision.routed_text(text, decision.facts_of(decision.route_facts(text, scenario, tools, context)))
            made = router.route(seen, scenario)
            try:
                save_correction(scenario, seen, route, {"route": made.get("clm_route", made["route"]),
                                                        "confidence": made["confidence"]})
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


def http_app(server: MCPServer, key: str, host: str = "127.0.0.1", allowed_hosts: list[str] | None = None):
    """The server over Streamable HTTP at /mcp, answering only requests that send `Authorization: Bearer <key>`."""
    security = TransportSecuritySettings(allowed_hosts=allowed_hosts) if allowed_hosts else None
    app = server.streamable_http_app(stateless_http=True, host=host, transport_security=security)
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


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code. Messages go to stderr: stdout is MCP's."""
    parser = argparse.ArgumentParser(description="Offer a scenario's decisions to other agents over MCP.")
    parser.add_argument("scenario", help="path to the scenario file")
    parser.add_argument("--review", action="store_true", help="also offer correct and agreement")
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--key-env", default=KEY_ENV, help="variable holding the key callers must send")
    parser.add_argument("--allowed-host", action="append", dest="allowed_hosts",
                        help="a Host header to accept, such as decisions.example.org:443 (repeatable)")
    parser.add_argument("--trace", metavar="FILE", help="write one line per call to FILE, saying where the time went")
    parser.add_argument("--trace-text", action="store_true", help="include each request's text in the trace")
    args = parser.parse_args(argv)
    key = os.environ.get(args.key_env, "").strip()
    if args.http and not key:
        print(f"error: set {args.key_env} to the key callers must send", file=sys.stderr)
        return 2
    try:
        scenario, router, tools = load(args.scenario)
    except (ScenarioError, ClmError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    trace = tracing.TraceWriter(args.trace, text=args.trace_text) if args.trace else None
    server = build_server(scenario, router, tools, args.review, trace=trace)
    if not args.http:
        server.run("stdio")
        return 0
    import uvicorn  # pylint: disable=import-outside-toplevel
    print(f"Serving MCP at http://{args.host}:{args.port}/mcp", file=sys.stderr)
    uvicorn.run(http_app(server, key, args.host, args.allowed_hosts), host=args.host, port=args.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
