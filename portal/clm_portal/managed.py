"""The portal's own decision server, and the jobs that change an environment, one at a time.

The portal starts one `clm-server` of its own, on 127.0.0.1 at a free port, with a key made at each start and given
only to that process. It never touches the server agents use. Jobs (`clm-load-documents`, `clm-derive`) run in a
worker thread, so a long one blocks no page; one lock covers them and every derived-layer write.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import anyio.to_thread
from mcp import Client

from clm_decision.connection import http_streams

SERVER_KEY_ENV = "CLM_PORTAL_SERVER_KEY"
READ_TIMEOUT = 600      # seconds to wait for a tool's reply, which a local model can take a while to give
PORTAL_KEY_ENV = "CLM_PORTAL_KEY"
ADMIT_ALL = ["--admit-derived", "proposed,checked,approved"]
ERROR_TAIL = 2000
_SECRET_WORDS = ("KEY", "PASSWORD", "TOKEN", "SECRET")


def script(name: str) -> Path | None:
    """A console script installed beside the portal's Python, or None when it is not installed."""
    found = Path(sys.executable).parent / name
    return found if found.is_file() else None


def child_environment(**more: str) -> dict:
    """The portal's environment for a child process, without the portal's own key."""
    return {**{name: value for name, value in os.environ.items() if name != PORTAL_KEY_ENV}, **more}


def scrub(text: str, *also: str) -> str:
    """The text with the value of every secret-looking variable, and of `also`, replaced by ***."""
    secrets_seen = [value for name, value in os.environ.items()
                    if any(word in name.upper() for word in _SECRET_WORDS) and len(value) >= 6]
    for value in sorted({*secrets_seen, *(item for item in also if item)}, key=len, reverse=True):
        text = text.replace(value, "***")
    return text


def free_port() -> int:
    """A port on 127.0.0.1 that nothing is listening on now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def connect(url: str, key: str):
    """Streams to an MCP server over HTTP, sending the key, through the SDK's own HTTP client as clm-agent does."""
    return http_streams(url, {"Authorization": f"Bearer {key}"}, read=READ_TIMEOUT)


class ToolFailed(RuntimeError):
    """The portal's server answered a call with an error; the message is the server's."""


class NotAnswering(ToolFailed):
    """The portal's server could not be reached, as while it restarts."""


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
        raise NotAnswering(f"the portal's server is not answering (it may be restarting): "
                           f"{found or type(found).__name__}") from error


class ManagedServer:  # pylint: disable=too-many-instance-attributes
    """The portal's decision server: started, restarted after every change, and stopped with the portal."""

    def __init__(self, environment: Path, *, popen=subprocess.Popen, connect=connect,  # pylint: disable=redefined-outer-name
                 wait: float = 120, find=script):
        self.environment, self.popen, self.connect, self.wait, self.find = Path(environment), popen, connect, wait, find
        self.testing, self.wanted, self.error = False, False, None
        self.process, self.url, self._key, self._stderr = None, None, None, None

    @property
    def running(self) -> bool:
        """Whether the server started and is still running. One that exited after starting gets its stderr, or its
        exit code, as `error`."""
        process = self.process
        if process is None or self.error is not None:
            return False
        code = process.poll()
        if code is not None:
            self.error = f"clm-server stopped after starting: {self._tail() or f'it exited with code {code}'}"
        return code is None

    def command(self, server: Path, port: int, testing: bool) -> list[str]:
        """The server's command line."""
        return [str(server), str(self.environment), "--http", "--host", "127.0.0.1",
                "--port", str(port), "--key-env", SERVER_KEY_ENV, *(ADMIT_ALL if testing else [])]

    async def start(self, testing: bool = False) -> None:
        """Start the server, and wait until it lists its tools or exits; `error` says why it did not start, and
        `testing` is true only once a server in testing mode has started. A start that fails leaves later restarts
        out of testing mode."""
        self.testing, self.wanted, self.error = False, False, None
        server = self.find("clm-server")
        if server is None:
            self.process, self.error = None, "clm-server is not installed in the portal's environment"
            return
        port, self._key = free_port(), secrets.token_urlsafe(32)
        self.url = f"http://127.0.0.1:{port}/mcp"
        if self._stderr is not None:
            self._stderr.close()
        self._stderr = tempfile.TemporaryFile()  # pylint: disable=consider-using-with
        try:
            self.process = self.popen(self.command(server, port, testing),
                                      env=child_environment(**{SERVER_KEY_ENV: self._key}),
                                      stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self._stderr)
        except OSError as error:
            self.process, self.error = None, f"cannot start clm-server: {error}"
            return
        deadline = time.monotonic() + self.wait
        while True:
            if self.process.poll() is not None:
                self.error = self._tail() or f"clm-server exited with code {self.process.poll()}"
                return
            try:
                await self.tools()
                self.testing = self.wanted = testing
                return
            except Exception:  # pylint: disable=broad-exception-caught
                if time.monotonic() > deadline:
                    self.error = self._tail() or f"clm-server did not answer within {self.wait:.0f} seconds"
                    await anyio.to_thread.run_sync(self.stop)
                    return
                await asyncio.sleep(0.2)

    async def restart(self, testing: bool | None = None) -> None:
        """Stop the server and start it again, in testing mode or not (unchanged when None). The wait for the old
        server to stop runs in a thread, so other pages are served meanwhile."""
        wanted = self.wanted if testing is None else testing
        await anyio.to_thread.run_sync(self.stop)
        await self.start(wanted)

    def stop(self) -> None:
        """Stop the server, if it is running; this waits up to 10 seconds for it to stop. The process is let go
        first, so a page that reads `running` meanwhile sees it stopped, not stopped with an error."""
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def _tail(self) -> str:
        self._stderr.seek(0)
        text = self._stderr.read().decode(errors="replace")
        return scrub(text[-ERROR_TAIL:], self._key or "").strip()

    def client(self) -> Client:
        """An MCP client of the server, to use as `async with`."""
        return Client(self.connect(self.url, self._key))

    async def tools(self) -> list:
        """The server's tools, as MCP lists them; NotAnswering when it cannot be reached."""
        async with answering():
            async with self.client() as client:
                return (await client.list_tools()).tools

    async def call(self, name: str, arguments: dict) -> dict:
        """One tool's result as a dict; ToolFailed with the server's message when it fails, NotAnswering when it
        cannot be reached."""
        async with answering():
            async with self.client() as client:
                return result_dict(name, await client.call_tool(name, arguments))


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


class Busy(RuntimeError):
    """A job or a change is already running; the message asks the person to wait."""


@dataclass
class Job:
    """One command the portal ran or is running, and what it printed."""
    name: str
    argv: list[str]
    running: bool = True
    returncode: int | None = None
    output: str = ""


class Jobs:
    """The jobs, one at a time; `lock` is also held for every derived-layer write."""

    def __init__(self, runner=subprocess.run, script=script):  # pylint: disable=redefined-outer-name
        self.runner, self.find = runner, script
        self._lock: asyncio.Lock | None = None
        self._current: Job | None = None
        self._last: Job | None = None
        self._task: asyncio.Task | None = None

    @property
    def lock(self) -> asyncio.Lock:
        """The lock jobs and layer writes share, made in the running event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def current(self) -> Job | None:
        """The job running now, if any."""
        return self._current

    def last(self) -> Job | None:
        """The job that ran most recently, running or not."""
        return self._last

    async def start(self, name: str, argv: list[str], on_success: Callable[[], Awaitable[None]] | None = None,
                    prepare: Callable[[], object] | None = None) -> Job:
        """Start a job in the background; Busy when one is running or a change is being written. A command that is
        not installed gives a finished job saying so. `prepare`, such as writing an upload, runs once the job holds
        the lock and before the command; when it raises, no job runs and the lock is freed."""
        if self.lock.locked():
            raise Busy("another job or change is running; wait for it to finish, then try again")
        found = self.find(argv[0])
        if found is None:
            job = Job(name, argv, running=False, output=f"{argv[0]} is not installed in the portal's environment")
            self._last = job
            return job
        await self.lock.acquire()
        if prepare is not None:
            try:
                prepare()
            except BaseException:
                self.lock.release()
                raise
        job = Job(name, argv)
        self._current = self._last = job
        self._task = asyncio.create_task(self._run(job, [str(found), *argv[1:]], on_success))
        return job

    async def _run(self, job: Job, argv: list[str], on_success) -> None:
        try:
            try:
                done = await anyio.to_thread.run_sync(lambda: self.runner(
                    argv, env=child_environment(), capture_output=True, text=True, stdin=subprocess.DEVNULL,
                    check=False))
                job.returncode, job.output = done.returncode, scrub((done.stdout or "") + (done.stderr or ""))
            except OSError as error:
                job.output = f"cannot run {job.argv[0]}: {error}"
            if job.returncode == 0 and on_success is not None:
                await on_success()
        finally:
            job.running = False
            self._current = None
            self.lock.release()

    async def wait(self) -> Job | None:
        """Wait for the running job, if any, to finish; the last job."""
        if self._task is not None:
            await self._task
        return self._last
