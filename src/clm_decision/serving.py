"""Which tools a server offers, and whether it is read-only: the file's [serve] table, replaced setting by setting by
clm-server's --tools, --exclude-tools and --read-only (docs/serving.md).

A name ending in * matches every tool starting with the rest, the framework's tool families (query_*, search_*).
Exclusion wins over inclusion, and describe is always offered, since an agent starts with it. A name or pattern that
matches no tool stops start-up with one line naming it and the tools there are.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .environment import _read  # one way to read either file, with the same one-line errors
from .scenario import ScenarioError

ALWAYS = "describe"
_KEYS = ("tools", "exclude_tools", "read_only")


@dataclass(frozen=True)
class Serve:
    """`tools` None offers every tool; `exclude_tools` removes some; `read_only` runs no acting tool."""
    tools: tuple[str, ...] | None = None
    exclude_tools: tuple[str, ...] = ()
    read_only: bool = False


def _names(table: dict, key: str) -> tuple[str, ...] | None:
    if key not in table:
        return None
    value = table[key]
    if not isinstance(value, list) or not value or not all(isinstance(name, str) and name for name in value):
        raise ScenarioError(f"serve.{key}: a list of tool names, such as [\"describe\", \"search_*\"]")
    return tuple(value)


def from_file(data: dict) -> Serve:
    """The [serve] table of a scenario or environment file, read as TOML; the default when there is none."""
    table = data.get("serve", {})
    if not isinstance(table, dict):
        raise ScenarioError("serve: must be a table")
    extra = sorted(set(table) - set(_KEYS))
    if extra:
        raise ScenarioError(f"serve: unknown setting {', '.join(extra)}; only tools, exclude_tools and read_only")
    read_only = table.get("read_only", False)
    if not isinstance(read_only, bool):
        raise ScenarioError("serve.read_only: true or false")
    return Serve(_names(table, "tools"), _names(table, "exclude_tools") or (), read_only)


def read(path: str | Path) -> Serve:
    """The [serve] table of the scenario or environment file at `path`."""
    return from_file(_read(Path(path)))


def _flag(value: str | None, flag: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise ScenarioError(f"{flag}: names no tool")
    return names


def merged(file: Serve, tools: str | None, exclude_tools: str | None, read_only: bool) -> Serve:
    """The file's settings with each one given on the command line replacing it; --read-only only turns it on."""
    chosen, excluded = _flag(tools, "--tools"), _flag(exclude_tools, "--exclude-tools")
    return Serve(file.tools if chosen is None else chosen, file.exclude_tools if excluded is None else excluded,
                 file.read_only or read_only)


def _matches(pattern: str, name: str) -> bool:
    return name.startswith(pattern[:-1]) if pattern.endswith("*") else name == pattern


def _check(patterns: tuple[str, ...], names: list[str], where: str) -> None:
    for pattern in patterns:
        if not any(_matches(pattern, name) for name in names):
            raise ScenarioError(f"{where}: '{pattern}' matches no tool; the tools are: {', '.join(names)}")


def choose(serve: Serve, names: list[str]) -> tuple[list[str], list[str]]:
    """-> (the tools offered, the tools left out), each in the order of `names`; raises ScenarioError."""
    if ALWAYS in serve.exclude_tools:
        raise ScenarioError(f"exclude_tools: {ALWAYS} is always offered, since an agent starts with it")
    _check(serve.tools or (), names, "tools")
    _check(serve.exclude_tools, names, "exclude_tools")
    offered = [name for name in names if name == ALWAYS or (
        (serve.tools is None or any(_matches(pattern, name) for pattern in serve.tools))
        and not any(_matches(pattern, name) for pattern in serve.exclude_tools))]
    return offered, [name for name in names if name not in offered]
