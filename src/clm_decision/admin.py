"""The tools the operator disables on a running environment server, and the middleware that hides and refuses them.

A disabled tool stays registered: the middleware takes it out of each tools/list result and refuses a tools/call for
it before the SDK runs it. The disabled names persist in disabled_tools.json beside the environment file, so they
survive a reload and a restart; they hide only tools the server offers, after the --tools selection, and a name that
a reload takes away stays in the file, since another reload may bring the tool back. describe cannot be disabled: an
agent reads it first.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from mcp.server.context import CallNext, HandlerResult, ServerMiddleware, ServerRequestContext
from mcp.server.subscriptions import InMemorySubscriptionBus, ToolsListChanged
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS

from .scenario import ScenarioError

FILE = "disabled_tools.json"
REFUSAL = "tool disabled by the operator"


class Admin:
    """The disabled tools, persisted at `path`, and the bus on which a change to the tool list is announced. `known`,
    set by the server it is given to, returns the tools offered now; until it is set, any name but describe is
    accepted."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.bus = InMemorySubscriptionBus()
        self.known: Callable[[], set[str]] | None = None
        self._disabled = self._load()
        self._publishing: set[asyncio.Task] = set()

    @classmethod
    def for_environment(cls, environment_path: Path) -> Admin:
        """The Admin whose file is beside the environment file."""
        return cls(Path(environment_path).parent / FILE)

    def _load(self) -> frozenset[str]:
        """The names in the file but describe, which cannot be disabled: none when there is no file, and a
        ScenarioError naming it when it cannot be read or is not {"disabled": [names]}."""
        if not self.path.exists():
            return frozenset()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ScenarioError(f"cannot read the disabled tools in {self.path}: {error}") from error
        names = data.get("disabled") if isinstance(data, dict) else None
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ScenarioError(f"{self.path} must be {{\"disabled\": [tool names]}}")
        return frozenset(names) - {"describe"}

    @property
    def disabled(self) -> frozenset[str]:
        """Every name in the file, offered now or not."""
        return self._disabled

    def hidden(self) -> frozenset[str]:
        """The disabled tools the server offers now: those the middleware hides and refuses. A name the server does
        not offer is left to the SDK, which says it has no such tool."""
        if self.known is None:
            return self._disabled
        return self._disabled & set(self.known())  # pylint: disable=not-callable  # None is ruled out above

    def _check(self, name: str) -> None:
        if name == "describe":
            raise ValueError("describe cannot be disabled: an agent reads it first")
        if self.known is None or name in self._disabled:
            return
        known = set(self.known())  # pylint: disable=not-callable  # set by the server; None is ruled out above
        if name not in known:
            raise ValueError(f"no tool '{name}'; the tools are: {', '.join(sorted(known))}")

    def disable(self, name: str) -> None:
        """Disable the tool `name` and save the set; a ValueError for describe or a tool the server lacks."""
        self._check(name)
        self._save(self._disabled | {name})

    def enable(self, name: str) -> None:
        """Enable the tool `name` again and save the set; a ValueError for describe or a tool the server lacks."""
        self._check(name)
        self._save(self._disabled - {name})

    def _save(self, names: frozenset[str]) -> None:
        """Write the set owner-only, through a temporary file replaced in one step, then hold it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)  # made 0600
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump({"disabled": sorted(names)}, file)
                file.write("\n")
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        self._disabled = names

    async def changed(self) -> None:
        """Announce that the tool list changed, to clients listening on the bus."""
        await self.bus.publish(ToolsListChanged())

    def announce(self) -> None:
        """changed(), scheduled on the running event loop; nothing when no loop runs on this thread."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.changed())
        self._publishing.add(task)      # held until done, since the loop keeps only a weak reference
        task.add_done_callback(self._publishing.discard)


def middleware(admin: Admin) -> ServerMiddleware[Any]:
    """Takes the disabled tools out of each tools/list result and refuses a tools/call for one. It traces nothing."""

    async def admin_middleware(ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
        disabled = admin.hidden() if ctx.method in ("tools/call", "tools/list") else frozenset()
        if ctx.method == "tools/call" and disabled and (ctx.params or {}).get("name") in disabled:
            raise MCPError(INVALID_PARAMS, REFUSAL)
        result = await call_next(ctx)
        if ctx.method == "tools/list" and disabled and isinstance(result, dict):
            result = {**result, "tools": [tool for tool in result.get("tools", []) if tool.get("name") not in disabled]}
        return result
    return admin_middleware
