"""The running server's checks, a short list the portal's Server page shows: CLM, PostgreSQL, Neo4j, OAuth, the
trace, the tool list's fingerprint and one synthetic describe.

Each check is pass, warn, fail or not configured (the server does not use that part). A check that touches a
database runs it in a thread, given 5 s to answer. A check of a part names the offered tools whose infrastructure
includes it (environment_server.infrastructure), so a failure says which tools it affects. The fingerprint is
SHA-256 over the names and input schemas of every registered tool, disabled ones too, against a baseline taken when
the checks are made and again at each reload, so neither a disable nor a reload reads as a vanished tool.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import anyio
import anyio.to_thread
from mcp import Client
from mcp.server import MCPServer

from . import tracing
from .environment import BuiltinConnector, Environment
from .environment_server import infrastructure
from .oauth import OidcVerifier

TIMEOUT_SECONDS = 5
PASS, WARN, FAIL, NOT_CONFIGURED = "pass", "warn", "fail", "not configured"
NAMES = ("clm", "postgres", "neo4j", "oauth", "trace", "tools", "describe")


@dataclass
class Check:  # pylint: disable=too-many-instance-attributes
    """One check's latest result. measured_at and last_success are times (time.time()); last_success is the last
    run that did not fail, last_error the detail of the last that did; both are kept across runs. tools are the
    offered tools that run on what the check checks."""
    name: str
    status: str
    detail: str
    measured_at: float
    last_success: float | None = None
    last_error: str | None = None
    tools: list[str] = field(default_factory=list)


def _first_line(error: BaseException) -> str:
    return (str(error).strip().splitlines() or [""])[0]


def _failure(error: BaseException) -> str:
    """An error in one line, with its type unless it is one of ours, whose message already says what failed."""
    line, kind = _first_line(error), type(error).__name__
    if kind in ("StepError", "OAuthError", "ScenarioError", "_Hung") and line:
        return line
    return f"{kind}: {line}" if line else kind


def _registered(server: MCPServer) -> dict[str, str]:
    """Every registered tool's input schema as JSON with sorted keys, by name: what list_tools gives, read straight
    from the registry, so it needs no event loop and no middleware filters it."""
    tools = server._tool_manager.list_tools()  # pylint: disable=protected-access
    return {tool.name: json.dumps(tool.parameters, sort_keys=True) for tool in tools}


def _fingerprint(schemas: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sorted(schemas.items())).encode()).hexdigest()


class _Hung(Exception):
    """A check's last ping is still running."""


class Checks:
    """The checks of a server built by build_environment_server, with its trace writer and OAuth verifier (None
    when it has none). Made once per server: it registers on server.offering, so reload moves the baseline."""

    def __init__(self, server: MCPServer, trace: tracing.TraceWriter | None, verifier: OidcVerifier | None):
        self.server, self.trace, self.verifier = server, trace, verifier
        self._results: dict[str, Check] = {}
        self._schemas: dict[str, str] = {}
        # The checks whose ping is still running in a thread, with when it started: one at a time each, so a hung
        # database never gathers threads
        self._pinging: dict[str, float] = {}
        self.baseline = ""
        self.rebaseline()
        offering = getattr(server, "offering", None)
        if offering is not None:
            offering.checks = self

    def rebaseline(self) -> None:
        """Takes the tools registered now as the expected ones."""
        self._schemas = _registered(self.server)
        self.baseline = _fingerprint(self._schemas)

    @property
    def latest(self) -> list[Check]:
        """The last results, in order; empty before the first run."""
        return [self._results[name] for name in NAMES if name in self._results]

    async def run(self, refresh: bool = False) -> list[Check]:
        """Runs every check, in order; with `refresh`, the OAuth check fetches the keys first."""
        offering = getattr(self.server, "offering", None)
        if offering is None:
            for name in NAMES[:3]:
                self._record(name, NOT_CONFIGURED, "", {})
        else:
            with offering.holder.use() as env:
                uses = {name: needs for name, needs in infrastructure(env).items() if name in offering.offered}
                # Each result is kept as soon as it is measured, so a check that never ends hides none before it
                self._record("clm", *await self._guarded(lambda: self._clm(env)), uses)
                self._record("postgres", *await self._guarded(lambda: self._postgres(env)), uses)
                self._record("neo4j", *await self._guarded(lambda: self._neo4j(env)), uses)
        self._record("oauth", *await self._guarded(lambda: self._oauth(refresh)), {})
        self._record("trace", *await self._guarded(self._trace), {})
        self._record("tools", *await self._guarded(self._tools), {})
        if self._results["postgres"].status == FAIL:
            # describe reads the store when document tools are offered, in a thread the SDK cannot abandon
            self._record("describe", FAIL, "skipped: PostgreSQL not answering", {})
        else:
            self._record("describe", *await self._guarded(self._describe), {})
        return self.latest

    async def _ping(self, name: str, ping: Callable[[], object]) -> object:
        """ping() in a thread, given TIMEOUT_SECONDS; raises TimeoutError past it, leaving the thread to finish. While
        the last ping of `name` has not returned, no other starts: that raises _Hung saying since when."""
        if name in self._pinging:
            since = time.strftime("%H:%M:%S", time.localtime(self._pinging[name]))
            raise _Hung(f"the last ping has not answered since {since}")
        self._pinging[name] = time.time()

        def run() -> object:
            try:
                return ping()
            finally:
                self._pinging.pop(name, None)
        with anyio.fail_after(TIMEOUT_SECONDS):
            return await anyio.to_thread.run_sync(run, abandon_on_cancel=True)

    def _record(self, name: str, status: str, detail: str, uses: dict[str, set[str]]) -> None:
        tools = sorted(tool for tool, needs in uses.items() if name in needs)
        before = self._results.get(name)
        now = time.time()
        last_success = before.last_success if before else None
        last_error = before.last_error if before else None
        if status == FAIL:
            last_error = detail
        else:
            last_success = now
        self._results[name] = Check(name, status, detail, now, last_success, last_error, tools)

    @staticmethod
    async def _guarded(check: Callable[[], Awaitable[tuple[str, str]]]) -> tuple[str, str]:
        """A check's result; an error it raises, or no answer in time, fails it in one line."""
        try:
            return await check()
        except TimeoutError:
            return FAIL, f"no answer within {TIMEOUT_SECONDS} s"
        # A check must report whatever goes wrong, never stop the others or the timer that runs them
        except Exception as error:  # pylint: disable=broad-exception-caught
            return FAIL, _failure(error)

    @staticmethod
    async def _clm(env: Environment) -> tuple[str, str]:
        if env.clm is None:
            return NOT_CONFIGURED, "no [clm] table"
        if env.clm._engine is None:  # pylint: disable=protected-access
            return WARN, "loads on first judgement"
        return PASS, f"loaded, on {env.clm.backend}"

    async def _postgres(self, env: Environment) -> tuple[str, str]:
        if env.store is None:
            return NOT_CONFIGURED, "no document store"
        reconnected = await self._ping("postgres", env.store.ping)
        return PASS, "SELECT 1 answered" + (" after reconnecting" if reconnected else "")

    async def _neo4j(self, env: Environment) -> tuple[str, str]:
        if isinstance(env.connector, BuiltinConnector):
            return NOT_CONFIGURED, "the built-in graph"
        await self._ping("neo4j", env.connector.ping)
        return PASS, "RETURN 1 answered"

    async def _oauth(self, refresh: bool) -> tuple[str, str]:
        verifier = self.verifier
        if verifier is None:
            return NOT_CONFIGURED, "no OAuth"
        if refresh:
            with anyio.fail_after(TIMEOUT_SECONDS):
                await verifier.refresh()
        if verifier.last_error is not None:
            return FAIL, verifier.last_error
        if verifier.fetched_at is None:
            return WARN, "no keys fetched yet"
        return PASS, f"keys {int(time.time() - verifier.fetched_at)} s old"

    async def _trace(self) -> tuple[str, str]:
        if self.trace is None:
            return NOT_CONFIGURED, "no --trace"
        if self.trace._stopped:  # pylint: disable=protected-access
            return FAIL, f"tracing stopped: cannot write to {self.trace.path}"
        try:
            mode = stat.S_IMODE(os.stat(self.trace.path).st_mode)
        except FileNotFoundError:
            # Opened at the first call traced
            return PASS, f"{self.trace.path}, not written yet"
        if mode != 0o600:
            return WARN, f"{self.trace.path} has mode {mode:04o}, not 0600: others can read it"
        return PASS, str(self.trace.path)

    async def _tools(self) -> tuple[str, str]:
        now = _registered(self.server)
        if _fingerprint(now) == self.baseline:
            return PASS, f"fingerprint {self.baseline[:16]}"
        vanished = sorted(set(self._schemas) - set(now))
        appeared = sorted(set(now) - set(self._schemas))
        changed = sorted(name for name in set(now) & set(self._schemas) if now[name] != self._schemas[name])
        parts = [f"{what}: {', '.join(names)}" for what, names in
                 (("vanished", vanished), ("appeared", appeared), ("schema changed", changed)) if names]
        return FAIL, "; ".join(parts)

    async def _describe(self) -> tuple[str, str]:
        """describe through the in-memory client, marked synthetic: the flag is set before the client starts the
        server's task, which copies it."""
        token = tracing.SYNTHETIC.set(True)
        try:
            with anyio.fail_after(TIMEOUT_SECONDS):
                async with Client(self.server) as client:
                    result = await client.call_tool("describe", {})
        finally:
            tracing.SYNTHETIC.reset(token)
        if result.is_error:
            text = next((part.text for part in result.content if hasattr(part, "text")), "describe failed")
            return FAIL, (text.strip().splitlines() or ["describe failed"])[0]
        return PASS, "describe answered"
