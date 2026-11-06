"""The decision server: a scenario's decisions offered to other agents as MCP tools, over stdio or HTTP; for an
environment over HTTP, with the portal (clm_decision.portal) on a second port of the same process."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import hmac
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.responses import JSONResponse

from . import decision
from . import oauth
from . import serving
from . import tracing
from .admin import Admin
from .clm_common import ClmError
from .evaluate import agreement as replay_agreement
from .dataset import labels
from .record import check_record, load_records, record_key, save_correction, skipped_note
from .environment import Environment, is_environment, load_environment
from .environment_server import build_environment_server
from .scenario import Scenario, ScenarioError, load_scenario
from .serving import MAX_TEXT, Serve
from .tools import check_tools, load_tools

KEY_ENV = "DECISION_SERVER_KEY"
MAX_BATCH = 1000
READ_ONLY = ToolAnnotations(read_only_hint=True)
CHANGES = ToolAnnotations(read_only_hint=False)
TOOLS = ("decide", "decide_many", "check_call", "describe")
REVIEW_TOOLS = ("correct", "agreement")
NO_REVIEW = "a read-only server cannot offer review mode: correct writes the record"
# With --allowed-host, still taken: a call from the server's own computer
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


def tool_names(review: bool) -> list[str]:
    """The tools a scenario's server can offer, before any selection."""
    return [*TOOLS, *(REVIEW_TOOLS if review else ())]


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


def _short(text: str, name: str = "text") -> None:
    """ToolError if a request's text is longer than CLM needs."""
    if len(text) > MAX_TEXT:
        raise ToolError(f"{name}: at most {MAX_TEXT} characters, not {len(text)}")


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


def _instructions(scenario: Scenario, offered: list[str], read_only: bool) -> str:
    """The server's instructions, naming only the deciding tool offered: decide, else decide_many, else neither."""
    start = f"Decisions for the scenario '{scenario.path.stem}'. Call describe to see the routes and questions"
    tool = next((name for name in ("decide", "decide_many") if name in offered), None)
    if tool is None:
        return f"{start}."
    what = "the text of a request" if tool == "decide" else "the texts of requests"
    acting = f"{tool} runs no acting tool: the server is read-only." if read_only else \
        f"{tool} may run the scenario's acting tools."
    return f"{start}, then {tool} with {what}. {acting}"


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def build_server(scenario: Scenario, router, tools: dict[str, Callable], review: bool,  # pylint: disable=too-many-arguments
                 trace: tracing.TraceWriter | None = None, auth: oauth.OAuth | None = None,
                 serve: Serve = Serve()) -> MCPServer:
    """The MCP server for one scenario. Review mode adds `correct` and `agreement`, which write and read the record.
    With `trace`, each call is written to the trace; with `auth`, HTTP callers need a token. `serve` chooses the
    tools offered and whether the server is read-only (serving.py); read-only with review is a ScenarioError."""
    if serve.read_only and review:
        raise ScenarioError(NO_REVIEW)
    offered, left_out = serving.choose(serve, tool_names(review), tool_names(True))
    server = MCPServer("decision-server", instructions=_instructions(scenario, offered, serve.read_only),
                       log_level="WARNING", **oauth.server_options(auth))
    deciding = READ_ONLY if serve.read_only else CHANGES
    # CLM runs on one model and calls arrive on worker threads, so its calls take turns
    router = _OneAtATime(router, threading.Lock())

    def clm(work: Callable[[], dict]) -> dict:
        try:
            return work()
        except ClmError as error:
            raise ToolError(str(error)) from error

    def traced(tool: str, text: str | list[str] | None = None):
        return trace.call(tool, text) if trace else contextlib.nullcontext({})

    def outcome(result: dict) -> dict:
        return {key: result.get(key) for key in ("route", "confidence", "uncertain", "source")}

    @server.tool(annotations=deciding)
    def decide(text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Decide on a request: its route and confidence, the answers to the scenario's questions, the rules
        that apply, facts the informing tools found, and what the acting tools did. An uncertain route runs no
        acting tool. `context` (such as a booking or user ID) is given to the scenario's tools, not to CLM. A text
        may be at most 32,000 characters."""
        _short(text)
        with traced("decide", text) as record:
            result = clm(lambda: decision.decide(text, scenario, router, tools, context, serve.read_only))
            if trace:
                record["outcome"] = outcome(result)
            return result

    @server.tool(annotations=deciding)
    def decide_many(requests: list[dict[str, Any]]) -> dict[str, Any]:
        """Decide on many requests at once, each {"text": ..., "context": ...} with context optional: decisions as
        decide makes them, in the same order, faster per request, except that every request's facts and options are
        read before any request's actions run. At most 1000 requests a call."""
        if len(requests) > MAX_BATCH:
            raise ToolError(f"decide_many takes at most {MAX_BATCH} requests a call, not {len(requests)}")
        for i, request in enumerate(requests):
            if not isinstance(request.get("text"), str) or not isinstance(request.get("context"), dict | None):
                raise ToolError(f"requests[{i}] must be {{'text': ..., 'context': {{...}}}}, with context optional")
            _short(request["text"], f"requests[{i}].text")
        with traced("decide_many", [request["text"] for request in requests]) as record:
            decisions = clm(lambda: decision.decide_many(requests, scenario, router, tools, serve.read_only))
            if trace:
                record["outcome"] = {"decisions": [outcome(made) for made in decisions]}
            return {"decisions": decisions}

    @server.tool(annotations=READ_ONLY)
    def check_call(text: str, call: dict[str, Any]) -> dict[str, Any]:
        """CLM's opinion of a tool call you propose for a request: a probability and whether it is approved."""
        _short(text)
        _short(json.dumps(call, sort_keys=True), "call")
        with traced("check_call", text) as record:
            result = clm(lambda: decision.check_call(text, call, scenario, router))
            record["outcome"] = result
            return result

    @server.tool(annotations=READ_ONLY)
    def describe() -> dict[str, Any]:
        """The decisions this server makes: its routes, questions and rules."""
        with traced("describe"):
            return {**decision.describe(scenario), **({"left_out": left_out} if left_out else {}),
                    **({"read_only": True} if serve.read_only else {})}

    if review:
        _add_review(server, scenario, router, tools, traced, clm)
    for name in tool_names(review):
        if name not in offered:
            server.remove_tool(name)
    return server


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _add_review(server: MCPServer, scenario: Scenario, router, tools: dict[str, Callable], traced: Callable,
                clm: Callable) -> None:
    """correct and agreement, which write and read the record."""

    @server.tool(annotations=CHANGES)
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def correct(text: str, route: str, context: dict[str, Any] | None = None, answers: dict[str, Any] | None = None,
                option: str | None = None) -> dict[str, Any]:
        """Review mode: save the right route for a request, for training. Near-identical requests get it at once.
        Send the same `context` as with decide, so the route is saved with the facts it was decided on.
        Optionally also the right `answers` to the questions (true/false, a level's number, or a choice's key)
        and the right `option`'s id, on a route with options."""
        _short(text)

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


def _without_port(host: str) -> str:
    """decisions.example.org:443, decisions.example.org:* and [::1]:8000 without their ports."""
    name, colon, port = host.rpartition(":")
    if colon and (port == "*" or port.isdigit()) and (":" not in name or name.endswith("]")):
        return name
    return host


def _security(allowed_hosts: list[str]) -> TransportSecuritySettings:
    """The Host and Origin headers to accept: each allowed host on any port or none (an HTTPS client leaves out
    443), a page served over https from it, and this computer's own addresses, as the SDK takes them by default."""
    names = list(dict.fromkeys([*map(_without_port, allowed_hosts), *LOCAL_HOSTS]))
    origins = []
    for name in names:
        for scheme in ("https", "http") if name in LOCAL_HOSTS else ("https",):
            origins += [f"{scheme}://{name}", f"{scheme}://{name}:*"]
    return TransportSecuritySettings(allowed_hosts=[form for name in names for form in (name, f"{name}:*")],
                                     allowed_origins=origins)


MIN_FILE_KEY = 16


class KeySet:
    """The keys agents may send, held as SHA-256 digests only. An agent's key is known by its id, the first 8 hex
    characters of its digest; `last_used` says when each id last opened a request."""

    def __init__(self, digests: list[str]):
        self._digests = list(dict.fromkeys(digests))
        self.last_used: dict[str, float] = {}

    @classmethod
    def from_sources(cls, env_key: str | None, file: Path | None) -> KeySet:
        """The key from a variable (any length) and each key in a file, one per line (blank lines and lines
        starting with # skipped, each key at least 16 characters); raises ScenarioError naming the file and line,
        never a key."""
        digests = [hashlib.sha256(env_key.encode()).hexdigest()] if env_key else []
        if file is not None:
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as error:
                raise ScenarioError(f"cannot read the key file {file}: {type(error).__name__}") from error
            for number, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if len(line) < MIN_FILE_KEY:
                    raise ScenarioError(f"{file} line {number}: a key must be at least {MIN_FILE_KEY} characters")
                digests.append(hashlib.sha256(line.encode()).hexdigest())
        if not digests:
            raise ScenarioError("no keys: give a key in the environment or a key file with at least one key")
        return cls(digests)

    def ids(self) -> list[str]:
        """The id of each key, in the order given."""
        return [digest[:8] for digest in self._digests]

    def match(self, sent: bytes) -> str | None:
        """The id of the key `sent` is, noting the time; None if it is none of them. Every digest is compared."""
        sent_digest = hashlib.sha256(sent).hexdigest().encode()
        found = None
        for digest in self._digests:
            if hmac.compare_digest(sent_digest, digest.encode()):
                found = digest[:8]
        if found is not None:
            self.last_used[found] = time.time()
        return found


def http_app(server: MCPServer, keys: KeySet | str | None, host: str = "127.0.0.1",
             allowed_hosts: list[str] | None = None):
    """The server over Streamable HTTP at /mcp, answering only requests that send `Authorization: Bearer <key>`
    (the scheme in any case) with one of the keys (a KeySet, or one key as text), and tracing each call under the
    key's id. With no keys, the server was built with OAuth and checks tokens itself."""
    security = _security(allowed_hosts) if allowed_hosts else None
    app = server.streamable_http_app(stateless_http=True, host=host, transport_security=security)
    if keys is None:
        return app
    if isinstance(keys, str):
        keys = KeySet.from_sources(keys, None)

    async def guarded(scope, receive, send):
        if scope["type"] == "http":
            scheme, _, sent = dict(scope["headers"]).get(b"authorization", b"").partition(b" ")
            key_id = keys.match(sent.strip()) if scheme.lower() == b"bearer" else None
            if key_id is None:
                await JSONResponse({"error": "missing or wrong key"}, status_code=401,
                                   headers={"WWW-Authenticate": 'Bearer realm="decision-server"'})(scope, receive, send)
                return
            token = tracing.CALLER_KEY_ID.set(key_id)
            try:
                await app(scope, receive, send)
            finally:
                tracing.CALLER_KEY_ID.reset(token)
            return
        await app(scope, receive, send)
    return guarded


def load(path: str):
    """-> (scenario, router, tools), with CLM loaded and earlier corrections ready to reuse; raises ScenarioError."""
    scenario = load_scenario(path)
    tools = load_tools(scenario)
    check_tools(scenario, tools)
    key = record_key(scenario)
    from .clm_router import make_router  # pylint: disable=import-outside-toplevel
    print("Loading CLM...", file=sys.stderr)
    router, note = make_router(scenario)
    print(f"CLM is using {note}.", file=sys.stderr)
    if scenario.live_corrections:
        records, skipped = load_records(scenario.record_path, key)
        if skipped:
            print(skipped_note(skipped, scenario.record_path), file=sys.stderr)
        if router.load_verdicts(records, scenario):
            print(f"Reusing {len(router.verdicts)} corrections from {scenario.record_path.name}.", file=sys.stderr)
    return scenario, router, tools


def load_any(path: str, admit: list[str] | None = None):
    """An environment, loaded without CLM, or a CLM scenario's (scenario, router, tools) as `load` gives them.
    `admit` overrides the derived statuses an environment admits, for testing only."""
    if is_environment(path):
        return load_environment(path, admit=admit)
    if admit is not None:
        raise ScenarioError(f"--admit-derived: {Path(path).name} is not an environment, so nothing derived to admit")
    return load(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offer a scenario's or an environment's decisions to other agents "
                                                 "over MCP.")
    parser.add_argument("scenario", metavar="FILE",
                        help="path to the scenario file, or to an environment's environment.toml")
    parser.add_argument("--review", action="store_true", help="also offer correct and agreement")
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--auth", choices=("key", "oauth"), default="key",
                        help="over HTTP: a shared key (default), or tokens from an OIDC issuer")
    parser.add_argument("--key-env", default=KEY_ENV, help="variable holding the key callers must send")
    parser.add_argument("--key-file", metavar="FILE", type=Path,
                        help="with --auth key: a file of keys callers may send, one per line (# comments, at least "
                             "16 characters each), besides the one in the variable")
    parser.add_argument("--issuer", help=f"with --auth oauth: the token issuer's URL (or {oauth.ISSUER_ENV})")
    parser.add_argument("--audience", help=f"with --auth oauth: the audience tokens must name (or "
                                           f"{oauth.AUDIENCE_ENV}; default: the resource URL)")
    parser.add_argument("--scope", action="append", dest="scopes",
                        help=f"with --auth oauth: a scope or app role tokens must carry (repeatable, or "
                             f"{oauth.SCOPES_ENV}, space-separated)")
    parser.add_argument("--resource-url", help="with --auth oauth: this server's public MCP address (default: "
                                               "http://HOST:PORT/mcp)")
    parser.add_argument("--allowed-host", action="append", dest="allowed_hosts",
                        help="a host callers name, such as decisions.example.org, taken on any port (repeatable)")
    parser.add_argument("--trace", metavar="FILE", help="write one line per call to FILE, saying where the time went")
    parser.add_argument("--trace-text", action="store_true",
                        help="include each request's text in the trace: decide, decide_many, check_call and "
                             "correct's; for an environment, each tool call's arguments")
    parser.add_argument("--tools", metavar="NAMES",
                        help="offer only these tools, comma-separated, and describe; NAME* matches a prefix "
                             "(replaces [serve] tools)")
    parser.add_argument("--exclude-tools", metavar="NAMES",
                        help="offer every tool but these, comma-separated; wins over --tools (replaces [serve] "
                             "exclude_tools)")
    parser.add_argument("--read-only", action="store_true",
                        help="run no acting tool: decide reports each one skipped (or [serve] read_only = true)")
    parser.add_argument("--admit-derived", metavar="STATUSES",
                        help="for testing only: admit derived rules and facts with these statuses, such as "
                             "proposed,checked,approved, not only approved ones; describe says so")
    portal = parser.add_argument_group("portal", "with --http and an environment: the administrator's page, on a "
                                                 "second port, signed in to with the key in CLM_PORTAL_KEY")
    portal.add_argument("--portal", metavar="PORT", type=int, help="serve the portal on this port")
    portal.add_argument("--portal-host", metavar="HOST", default="127.0.0.1",
                        help="the portal's address (default 127.0.0.1; any other is plain HTTP)")
    portal.add_argument("--portal-public-origin", metavar="URL",
                        help="the address a TLS-terminating proxy serves the portal at, such as "
                             "https://portal.example.org; a form may also come from it")
    portal.add_argument("--ask-backend", choices=("openai", "claude"), default="openai",
                        help="Ask's starting model, changeable on the page: openai-style (default) or claude")
    portal.add_argument("--ask-model", help="Ask's starting model name (default: the chat's for the backend)")
    portal.add_argument("--ask-model-url", metavar="URL",
                        help="Ask's model's OpenAI-style address (default: Ollama on this computer)")
    return parser


def _portal_settings(args: argparse.Namespace) -> tuple[str, str | None]:
    """The portal's sign-in key and public origin, checked before anything loads; ScenarioError saying what is
    wrong, never the key. Warns when the portal listens beyond this computer."""
    # pylint: disable=import-outside-toplevel
    from .portal import LOOPBACK, MIN_KEY, PORTAL_KEY_ENV, web
    if not args.http:
        raise ScenarioError("--portal needs --http: the portal is served beside the server's HTTP port")
    if not is_environment(args.scenario):
        raise ScenarioError("--portal is for environments; a CLM scenario has no documents or derived rules to "
                            "administer")
    if args.portal == args.port:
        raise ScenarioError(f"--portal {args.portal} is the server's own --port; give the portal another")
    key = os.environ.get(PORTAL_KEY_ENV, "").strip()
    if not key:
        raise ScenarioError(f"set {PORTAL_KEY_ENV} to the key that signs in to the portal; it is not the key agents "
                            "use")
    if len(key) < MIN_KEY:
        raise ScenarioError(f"{PORTAL_KEY_ENV} must be at least {MIN_KEY} characters; use a long random key")
    try:
        public = web.public_origin(args.portal_public_origin) if args.portal_public_origin else None
    except ValueError as error:
        raise ScenarioError(str(error).replace("--public-origin", "--portal-public-origin")) from error
    if args.portal_host not in LOOPBACK:
        print(f"warning: the portal listens on {args.portal_host}: it is plain HTTP, so its key and every page cross "
              "the network unencrypted; put it behind TLS or keep it on 127.0.0.1", file=sys.stderr)
    return key, public


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def _serve_with_portal(args: argparse.Namespace, server: MCPServer, admin: Admin, trace: tracing.TraceWriter | None,
                       auth: oauth.OAuth | None, keys: KeySet | None, portal_key: str, public: str | None) -> int:
    """Serve the agents' port and the portal's from one process until SIGINT or SIGTERM."""
    # pylint: disable=import-outside-toplevel
    from .checks import Checks
    from .portal import live, managed, web
    checks = Checks(server, trace, auth.verifier if auth else None)
    shown = f"[{args.host}]" if ":" in args.host else args.host
    portal = web.Portal(Path(args.scenario), live.LiveServer(server, admin, checks),
                        managed.Jobs(withheld=(args.key_env,)),
                        options={"backend": args.ask_backend, "model": args.ask_model, "model_url": args.ask_model_url},
                        agents_url=args.resource_url or f"http://{shown}:{args.port}/mcp", keys=keys)
    portal_app = web.create_app(portal, portal_key, args.portal_host, args.portal, public)
    print(f"Serving MCP at http://{shown}:{args.port}/mcp"
          + (f", taking tokens from {auth.settings.issuer_url}" if auth else "")
          + f"; the portal at http://{args.portal_host}:{args.portal}/, signed in to with the key in CLM_PORTAL_KEY",
          file=sys.stderr)
    live.serve_both(http_app(server, keys, args.host, args.allowed_hosts), portal_app, (args.host, args.port),
                    (args.portal_host, args.portal))
    return 0


# pylint: disable-next=too-many-return-statements,too-many-branches,too-many-locals
def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code. Messages go to stderr: stdout is MCP's."""
    args = _parser().parse_args(argv)
    keys, auth = None, None
    if args.key_file and args.auth == "oauth":
        print("error: --key-file is for --auth key; with --auth oauth callers send tokens", file=sys.stderr)
        return 2
    if args.http and args.auth == "oauth":
        try:
            auth = oauth.from_args(args.issuer, args.audience, args.scopes,
                                   args.resource_url or f"http://{args.host}:{args.port}/mcp")
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    elif args.http:
        key = os.environ.get(args.key_env, "").strip()
        if not key and not args.key_file:
            print(f"error: set {args.key_env} to the key callers must send, or give --key-file", file=sys.stderr)
            return 2
        try:
            keys = KeySet.from_sources(key or None, args.key_file)
        except ScenarioError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    try:
        if args.review and is_environment(args.scenario):
            print("error: review mode is for CLM scenarios; an environment has no corrections to review",
                  file=sys.stderr)
            return 2
        if args.admit_derived and not is_environment(args.scenario):
            print("error: --admit-derived is for environments; a CLM scenario has no derived rules", file=sys.stderr)
            return 2
        portal_key, public = _portal_settings(args) if args.portal is not None else (None, None)
        serve = serving.merged(serving.read(args.scenario), args.tools, args.exclude_tools, args.read_only)
        if serve.read_only and args.review:
            raise ScenarioError(NO_REVIEW)
        if not is_environment(args.scenario):
            # Checked before CLM loads, which takes a while
            serving.choose(serve, tool_names(args.review), tool_names(True))
        if args.review:
            # correct writes the record, so it must take this key; checked before CLM loads too
            check_record(load_scenario(args.scenario))
        # A disabled_tools.json that cannot be read stops start-up before the environment loads
        admin = Admin.for_environment(Path(args.scenario)) if is_environment(args.scenario) else None
        loaded = load_any(args.scenario, args.admit_derived.split(",") if args.admit_derived else None)
        trace = tracing.TraceWriter(args.trace, text=args.trace_text, read_only=serve.read_only) \
            if args.trace else None
        if isinstance(loaded, Environment):
            server = build_environment_server(loaded, trace=trace, auth=auth, serve=serve,
                                              admin=admin)
        else:
            scenario, router, tools = loaded
            server = build_server(scenario, router, tools, args.review, trace=trace, auth=auth, serve=serve)
    except (ScenarioError, ClmError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not args.http:
        server.run("stdio")
        return 0
    if args.portal is not None:
        return _serve_with_portal(args, server, admin, trace, auth, keys, portal_key, public)
    import uvicorn  # pylint: disable=import-outside-toplevel
    print(f"Serving MCP at http://{args.host}:{args.port}/mcp"
          + (f", taking tokens from {auth.settings.issuer_url}" if auth else ""), file=sys.stderr)
    uvicorn.run(http_app(server, keys, args.host, args.allowed_hosts), host=args.host, port=args.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
