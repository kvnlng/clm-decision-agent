"""The decision server: a scenario's decisions offered to other agents as MCP tools, over stdio or HTTP."""
from __future__ import annotations

import argparse
import dataclasses
import hmac
import os
import sys
import threading
from typing import Any, Callable

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.responses import JSONResponse

import decision
from evaluate import agreement as replay_agreement
from record import load_records, record_key, save_correction
from scenario import Scenario, ScenarioError, load_scenario
from tools import check_tools, load_tools

KEY_ENV = "DECISION_SERVER_KEY"
READ_ONLY = ToolAnnotations(read_only_hint=True)
CHANGES = ToolAnnotations(read_only_hint=False)


def build_server(scenario: Scenario, router, tools: dict[str, Callable], review: bool) -> MCPServer:
    """The MCP server for one scenario. Review mode adds `correct` and `agreement`, which write and read the record."""
    server = MCPServer("decision-server", instructions=(
        f"Decisions for the scenario '{scenario.path.stem}'. Call describe to see the routes and questions, "
        "then decide with the text of a request. decide may run the scenario's acting tools."))
    # CLM runs on one model; calls arrive on worker threads, so they take turns
    lock = threading.Lock()

    def clm(work: Callable[[], dict]) -> dict:
        from clm_router import ClmError  # pylint: disable=import-outside-toplevel
        with lock:
            try:
                return work()
            except ClmError as error:
                raise ToolError(str(error)) from error

    @server.tool(annotations=CHANGES)
    def decide(text: str) -> dict[str, Any]:
        """Decide on a request: its route and confidence, the answers to the scenario's questions, the rules
        that apply, facts the informing tools found, and what the acting tools did. An uncertain route runs no
        acting tool."""
        return clm(lambda: decision.decide(text, scenario, router, tools))

    @server.tool(annotations=READ_ONLY)
    def check_call(text: str, call: dict[str, Any]) -> dict[str, Any]:
        """CLM's opinion of a tool call you propose for a request: a probability and whether it is approved."""
        return clm(lambda: decision.check_call(text, call, scenario, router))

    @server.tool(annotations=READ_ONLY)
    def describe() -> dict[str, Any]:
        """The decisions this server makes: its routes, questions and rules."""
        return decision.describe(scenario)

    if not review:
        return server

    @server.tool(annotations=CHANGES)
    def correct(text: str, route: str) -> dict[str, Any]:
        """Review mode: save the right route for a request, for training. Near-identical requests get it at once."""
        def work() -> dict:
            made = router.decide(text, scenario)
            try:
                save_correction(scenario, text, route, {"route": made.get("clm_route", made["route"]),
                                                        "confidence": made["confidence"]})
            except ValueError as error:
                raise ToolError(str(error)) from error
            if scenario.live_corrections:
                router.add_verdict(text, route, scenario)
            return {"saved": True}
        return clm(work)

    @server.tool(annotations=READ_ONLY)
    def agreement() -> dict[str, Any]:
        """Review mode: how often CLM's own route agrees with the saved corrections."""
        records, _ = load_records(scenario.record_path, record_key(scenario))
        # Replayed without reusing corrections, or every correction would agree with itself
        plain = dataclasses.replace(scenario, live_corrections=False)
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
    args = parser.parse_args(argv)
    key = os.environ.get(args.key_env, "").strip()
    if args.http and not key:
        print(f"error: set {args.key_env} to the key callers must send", file=sys.stderr)
        return 2
    try:
        scenario, router, tools = load(args.scenario)
    except ScenarioError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    server = build_server(scenario, router, tools, args.review)
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
