"""The jobs that change an environment, one at a time, and what the portal keeps out of them.

Jobs (`clm-load-documents`, `clm-derive`) run as child processes from a worker thread, so a long one blocks no page;
one lock covers them and every derived-layer write. A child's environment holds neither the portal's key nor the key
agents send the server.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import anyio.to_thread

from . import PORTAL_KEY_ENV

_SECRET_WORDS = ("KEY", "PASSWORD", "TOKEN", "SECRET")


def script(name: str) -> Path | None:
    """A console script installed beside the portal's Python, or None when it is not installed."""
    found = Path(sys.executable).parent / name
    return found if found.is_file() else None


def child_environment(*withheld: str) -> dict:
    """The process's environment for a child, without the portal's key or the variables named in `withheld`."""
    left_out = {PORTAL_KEY_ENV, *withheld}
    return {name: value for name, value in os.environ.items() if name not in left_out}


def scrub(text: str, *also: str) -> str:
    """The text with the value of every secret-looking variable, and of `also`, replaced by ***."""
    secrets_seen = [value for name, value in os.environ.items()
                    if any(word in name.upper() for word in _SECRET_WORDS) and len(value) >= 6]
    for value in sorted({*secrets_seen, *(item for item in also if item)}, key=len, reverse=True):
        text = text.replace(value, "***")
    return text


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
    """The jobs, one at a time; `lock` is also held for every derived-layer write. `withheld` names the variables,
    besides the portal's key, a job's environment leaves out: the one holding the key agents send the server."""

    def __init__(self, runner=subprocess.run, script=script,  # pylint: disable=redefined-outer-name
                 withheld: tuple[str, ...] = ()):
        self.runner, self.find, self.withheld = runner, script, tuple(withheld)
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
                    argv, env=child_environment(*self.withheld), capture_output=True, text=True,
                    stdin=subprocess.DEVNULL, check=False))
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
