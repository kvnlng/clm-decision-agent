"""The server agents use, as the portal in the same process sees it, and the two ports they are served on.

The portal calls the server in process, through the SDK's in-memory transport, so Ask, Configuration and Review see
exactly what agents see. A change takes effect at once: `reload` reads the environment file again and swaps the new
environment in while calls keep running. Testing builds a second, private server that admits unapproved derived
rules; Ask talks to it while the agents' server never offers them.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import signal
from typing import Callable

import anyio.to_thread
import uvicorn
from mcp import Client
from mcp.server import MCPServer

from clm_decision import admin as admin_module
from clm_decision.environment import OVERRIDE_ADMITS, Environment, load_environment
from clm_decision.environment_server import build_environment_server, reload as reload_server

from .managed import scrub


class ToolFailed(RuntimeError):
    """The server answered a call with an error; the message is the server's."""


class NotAnswering(ToolFailed):
    """The server could not be reached."""


def cause(error: BaseException) -> BaseException:
    """The first error inside any exception groups, as the MCP client wraps them."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return error


@contextlib.asynccontextmanager
async def answering():
    """Turn a failure to reach the server, however the MCP client wraps it, into NotAnswering; a ToolFailed from
    inside stays as it is."""
    try:
        yield
    except ToolFailed:
        raise
    except Exception as error:  # pylint: disable=broad-exception-caught
        found = cause(error)
        if isinstance(found, ToolFailed):
            raise found from error
        raise NotAnswering(f"the server is not answering: {found or type(found).__name__}") from error


def result_dict(name: str, result) -> dict:
    """A tool's result as a dict; ToolFailed with the server's message when it failed."""
    text = "\n".join(block.text for block in result.content if block.type == "text")
    if result.is_error:
        raise ToolFailed(text or f"{name} failed")
    if result.structured_content is not None:
        return result.structured_content
    try:
        found = json.loads(text) if text else {}
    except json.JSONDecodeError as error:
        raise ToolFailed(f"{name} did not answer with JSON") from error
    return found if isinstance(found, dict) else {"result": found}


class LiveServer:  # pylint: disable=too-many-instance-attributes
    """The agents' server (built by build_environment_server), its admin and its checks, for the portal, and `keys`,
    the server.KeySet agents send (None under OAuth), which must never hold `portal_key`. `error` says why the last
    reload failed, while the server goes on serving what it loaded before; `key_error` why the last reload could not
    read the agents' keys again, while the keys from before go on opening; `testing` is whether Ask talks to the
    private testing server, and `testing_error` why testing was turned off when a reload that succeeded could not
    build it again. One lock takes reloads and testing changes in turn, so a load never reuses an environment a
    concurrent reload retired; registry changes run on the event loop."""

    running = True      # the server is built before the portal starts, and serves until the process stops

    # pylint: disable-next=too-many-arguments
    def __init__(self, server: MCPServer, admin: admin_module.Admin | None, checks, *, keys=None,
                 portal_key: str | None = None):
        self.server, self.admin, self.checks = server, admin, checks
        self.keys, self._portal_key = keys, portal_key
        self.error: str | None = None
        self.key_error: str | None = None
        self.testing_error: str | None = None
        self._testing: MCPServer | None = None
        self._testing_env: Environment | None = None
        self._lock: asyncio.Lock | None = None

    @property
    def testing(self) -> bool:
        """Whether Ask talks to the testing server."""
        return self._testing is not None

    @property
    def lock(self) -> asyncio.Lock:
        """The lock reloads and testing changes share, made in the running event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def client(self) -> Client:
        """An in-process MCP client of the agents' server, or of the testing server while testing, to use as
        `async with`."""
        return Client(self._testing or self.server)

    async def call(self, name: str, arguments: dict) -> dict:
        """One tool's result as a dict; ToolFailed with the server's message when it fails."""
        async with answering():
            async with self.client() as client:
                return result_dict(name, await client.call_tool(name, arguments))

    async def _load(self, admit: list[str] | None) -> Environment:
        """The environment file read again in a thread, sharing what the current environment holds open."""
        current = self.server.holder.current
        return await anyio.to_thread.run_sync(functools.partial(load_environment, current.path, admit=admit,
                                                                reuse=current))

    async def reload(self) -> None:
        """Read the environment file and the agents' key file again and serve them: calls already running finish
        with the old environment. Never raises: when the load or the choice of tools fails, the old environment goes
        on serving and `error` says why; a reload that succeeds clears it and builds the testing server again, if one
        is on. When that rebuild fails, the reload still succeeded: testing is turned off and `testing_error` says
        why. The keys are read whether or not the environment loads: when they cannot be, the keys from before go on
        opening and `key_error` says why, never showing a key."""
        async with self.lock:
            await self._reload_keys()
            current = self.server.holder.current
            try:
                new = await self._load(list(current.admit) if current.admit_override else None)
                reload_server(self.server, new)
            except Exception as error:  # pylint: disable=broad-exception-caught
                self.error = ("the last reload failed, so the server still serves what it loaded before: "
                              f"{scrub(str(error) or type(error).__name__)}")
                return
            self.error = None
            if self._testing is not None:
                try:
                    await self._build_testing()
                except Exception as error:  # pylint: disable=broad-exception-caught
                    self.testing_error = ("the change is served; testing was turned off because the testing server "
                                          f"could not be built: {scrub(str(error) or type(error).__name__)}")

    async def _reload_keys(self) -> None:
        if self.keys is None:
            return
        try:
            await anyio.to_thread.run_sync(functools.partial(self.keys.reload, forbidden=self._portal_key))
        except Exception as error:  # pylint: disable=broad-exception-caught
            self.key_error = ("the agents' keys were not read again, so the keys from before still open: "
                              f"{scrub(str(error) or type(error).__name__)}")
            return
        self.key_error = None

    async def set_testing(self, on: bool) -> None:
        """Point Ask at a private testing server, which admits every derived status but rejected, or back at the
        agents' server. The error of a testing server that cannot be built is raised, and testing stays off."""
        async with self.lock:
            self.testing_error = None
            if on:
                await self._build_testing()
            else:
                self._drop_testing()

    async def _build_testing(self) -> None:
        """Build the testing server on the file as it is now, replacing any before it; on failure testing is off."""
        try:
            env = await self._load(list(OVERRIDE_ADMITS))
        except BaseException:
            self._drop_testing()
            raise
        try:
            # The agents' admin hides the tools it disables here too; given as admin, it would follow this server
            hidden = {"middleware": [admin_module.middleware(self.admin)]} if self.admin else {}
            server = build_environment_server(env, serve=self.server.offering.serve, **hidden)
        except BaseException:
            env.close(keeping=self.server.holder.alive())
            self._drop_testing()
            raise
        old, self._testing, self._testing_env = self._testing_env, server, env
        if old is not None:
            old.close(keeping=[*self.server.holder.alive(), env])

    def _drop_testing(self) -> None:
        old, self._testing, self._testing_env = self._testing_env, None, None
        if old is not None:
            old.close(keeping=self.server.holder.alive())


class _Quiet(uvicorn.Server):
    """A uvicorn server that leaves signals to serve_both: uvicorn 0.54 has no install_signal_handlers setting, and
    each server would otherwise take SIGINT and SIGTERM for itself and raise them again on the way out."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


def _on_signal(stop: Callable[[], None]) -> Callable[[], None]:
    """SIGINT and SIGTERM call `stop`; returns what puts the handlers back."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop)
    return lambda: [loop.remove_signal_handler(sig) for sig in (signal.SIGINT, signal.SIGTERM)]


def serve_both(agents_app, portal_app, agents: tuple[str, int], portal: tuple[str, int],
               on_signal: Callable[[Callable[[], None]], Callable[[], None]] = _on_signal) -> None:
    """Serve the agents' app at agents (host, port) and the portal's at portal, on one event loop, until SIGINT or
    SIGTERM, or until either stops, stops both; returns once both have. `on_signal(stop)` installs what calls stop
    and returns what removes it."""
    asyncio.run(_serve_both([(agents_app, *agents), (portal_app, *portal)], on_signal))


async def _serve_both(apps: list[tuple], on_signal) -> None:
    servers = [_Quiet(uvicorn.Config(app, host=host, port=port, log_level="warning")) for app, host, port in apps]

    def stop() -> None:
        for server in servers:
            server.should_exit = True

    async def serve(server: uvicorn.Server) -> None:
        try:
            await server.serve()
        finally:
            stop()      # one that stops, as when its port is taken, stops the other
    remove = on_signal(stop)
    try:
        await asyncio.gather(*(serve(server) for server in servers))
    finally:
        remove()
