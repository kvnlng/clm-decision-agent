"""Optional tracing of the server's work: one JSON line per call to an MCP tool (server.py --trace).

Each line says how long the call took and where the time went: waiting for CLM, encoding text, the
heads, the scenario's tools, and matching stored corrections. The engines and the decision report into
the call being traced; outside a traced call, reporting does nothing. trace_report.py summarises a file.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import threading
import time
from pathlib import Path
from typing import Generator

# The record of the call being traced on this thread, or None
_CURRENT: contextvars.ContextVar[dict | None] = contextvars.ContextVar("trace", default=None)


def add(key: str, amount: float = 1) -> None:
    """Add `amount` to a count of the call being traced."""
    record = _CURRENT.get()
    if record is not None:
        record["counts"][key] = record["counts"].get(key, 0) + amount


@contextlib.contextmanager
def timed(key: str) -> Generator[None]:
    """Add the milliseconds the block takes to a count of the call being traced."""
    if _CURRENT.get() is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        add(key, (time.perf_counter() - start) * 1000)


def note_tool(name: str, kind: str, ms: float, status: str) -> None:
    """Note one run of a scenario tool: its name, "inform" or "act", how long it took and its status."""
    record = _CURRENT.get()
    if record is not None:
        record["tools"].append({"tool": name, "kind": kind, "ms": ms, "status": status})


class TraceWriter:
    """Appends one line per traced call to `path`. The request's text is left out unless `text` is True."""

    def __init__(self, path, text: bool = False):
        self.path, self.text = Path(path), text
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def call(self, tool: str, request_text: str | None = None) -> Generator[dict]:
        """Trace one call; the block may fill in the record's "outcome"."""
        record: dict = {"tool": tool, "started": time.time(), "counts": {}, "tools": [], "outcome": {}}
        if self.text and request_text is not None:
            record["text"] = request_text
        token = _CURRENT.set(record)
        start = time.perf_counter()
        try:
            yield record
        except Exception as error:
            record["outcome"] = {"error": type(error).__name__}
            raise
        finally:
            _CURRENT.reset(token)
            record["ms"] = (time.perf_counter() - start) * 1000
            line = json.dumps(record) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as file:
                file.write(line)
