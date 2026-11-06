"""The environment a server serves now, replaceable while calls are running.

Each call takes the current environment when it starts and keeps it to its end; `swap` puts a new one in place for
the calls that start after it. An environment swapped out is closed once no call holds it, and keeps open what any
environment still alive shares with it. Tools run on worker threads and a reload in another, so a lock guards the
state; nothing serialises the calls themselves.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Iterator

from .environment import Environment


class Holder:
    """The current environment, and how many calls hold each environment still alive."""

    def __init__(self, env: Environment):
        self._lock = threading.Lock()
        self._current = env
        self._counts: dict[int, int] = {}               # id(environment) -> calls holding it
        self._retired: dict[int, Environment] = {}      # swapped out, closed when its last call ends

    @property
    def current(self) -> Environment:
        """The environment a call starting now would use."""
        with self._lock:
            return self._current

    def _alive(self) -> list[Environment]:
        """The environments a call holds or can still take: the current one and the retired ones still in use.
        Called with the lock held."""
        return [self._current, *self._retired.values()]

    @contextlib.contextmanager
    def use(self) -> Iterator[Environment]:
        """The current environment, held until the block ends; an environment swapped out meanwhile is closed when
        its last call ends."""
        with self._lock:
            env = self._current
            self._counts[id(env)] = self._counts.get(id(env), 0) + 1
        try:
            yield env
        finally:
            closing, keeping = None, []
            with self._lock:
                left = self._counts[id(env)] - 1
                if left:
                    self._counts[id(env)] = left
                else:
                    del self._counts[id(env)]
                    closing = self._retired.pop(id(env), None)
                    keeping = self._alive()
            if closing is not None:
                closing.close(keeping=keeping)

    def swap(self, new: Environment) -> Environment:
        """Puts `new` in place and returns the environment it replaces, which is closed now if no call holds it and
        otherwise when its last call ends."""
        with self._lock:
            old = self._current
            if new is old:
                return old
            self._current = new
            if id(old) in self._counts:
                self._retired[id(old)] = old
                return old
            keeping = self._alive()
        old.close(keeping=keeping)
        return old
