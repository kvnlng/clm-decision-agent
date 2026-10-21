"""Optional tracing of the server's work: one JSON line per call to an MCP tool (server.py --trace).

Each line says how long the call took and where the time went: waiting for CLM, encoding text, the
heads, the scenario's tools, and matching stored corrections. The engines and the decision report into
the call being traced; outside a traced call, reporting does nothing. A background thread writes the
lines, so writing costs a call nothing and a trace that cannot be written never fails one.
trace_report.py summarises a file.
"""
from __future__ import annotations

import atexit
import contextlib
import contextvars
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Generator

# The record of the call being traced on this thread, or None
_CURRENT: contextvars.ContextVar[dict | None] = contextvars.ContextVar("trace", default=None)


def active() -> bool:
    """Whether a call is being traced, for counts that cost something to work out."""
    return _CURRENT.get() is not None


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
        self._records: queue.SimpleQueue = queue.SimpleQueue()
        self._stopped = False
        self._thread = threading.Thread(target=self._write, name="trace-writer", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def _write(self) -> None:
        """Write records as they come until close() sends None. On a failure, say so once and stop tracing."""
        file = None
        while (record := self._records.get()) is not None:
            if self._stopped:
                continue
            try:
                line = json.dumps(record) + "\n"
                # Opened once and kept open for the writer's life, so it is closed below, not by a with
                file = file or self.path.open("a", encoding="utf-8")  # pylint: disable=consider-using-with
                file.write(line)
                if self._records.empty():
                    file.flush()
            except (OSError, TypeError, ValueError) as error:
                self._stopped = True
                print(f"warning: tracing stopped: cannot write to {self.path}: {error}", file=sys.stderr)
        if file:
            file.close()

    def close(self) -> None:
        """Write what is left and stop. Called at exit too."""
        if self._thread.is_alive():
            self._records.put(None)
            self._thread.join()

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
            if not self._stopped:
                self._records.put(record)
