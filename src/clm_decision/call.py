"""Call a decision server's tools by hand, without an agent: list them, or call one and print its result.

    clm-call FILE_OR_URL                          the tools, with their arguments and descriptions
    clm-call FILE_OR_URL TOOL '{"name": value}'   one call; its result as JSON

A file (a scenario or an environment.toml) starts the server over stdio, as an agent would. A URL connects to a
running server over HTTP, sending the key in DECISION_SERVER_KEY when it is set. `connect` is shared with clm-agent.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import anyio
from mcp import Client, MCPError, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

KEY_ENV = "DECISION_SERVER_KEY"


def is_url(target: str) -> bool:
    """Whether the target is a running server's address rather than a file."""
    return target.startswith(("http://", "https://"))


def server_params(path: str) -> StdioServerParameters:
    """How to start the decision server for a file. It gets this process's variables, so a password the
    environment names (NEO4J_PASSWORD) reaches it; the MCP client would otherwise pass on only a few, such as PATH."""
    return StdioServerParameters(command=sys.executable, args=["-m", "clm_decision.server", path],
                                 env=dict(os.environ))


@contextlib.asynccontextmanager
async def _http(url: str, refused: list[int]):
    key = os.environ.get(KEY_ENV)
    headers = {"Authorization": f"Bearer {key}"} if key else None
    async with create_mcp_http_client(headers=headers) as http, \
            streamable_http_client(url, http_client=http) as streams:
        # The SDK reports a refusal only as "an error response"; keep its status to say why
        async def note(response) -> None:
            if response.status_code in (401, 403):
                refused.append(response.status_code)
        http.event_hooks["response"].append(note)
        yield streams


def connect(target: str, refused: list[int] | None = None):
    """What to give mcp.Client for a file (started over stdio) or a URL (over HTTP, with the key if set). A list
    given as `refused` collects the status of any response refusing the key."""
    return _http(target, refused if refused is not None else []) if is_url(target) else server_params(target)


def _cause(error: BaseException) -> str:
    """The innermost error of a group, as one line."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return str(error) or type(error).__name__


def refusal(refused: list[int]) -> str | None:
    """Why the server refused the caller, if it did."""
    if not refused:
        return None
    sent = "with no key" if not os.environ.get(KEY_ENV) else f"with the key in {KEY_ENV}"
    return f"the server refused the call ({refused[0]}), {sent}"


async def _run(args, arguments: dict, refused: list[int]) -> int:
    async with Client(connect(args.target, refused)) as client:
        if not args.tool:
            for tool in (await client.list_tools()).tools:
                print(f"{tool.name}({', '.join(tool.input_schema.get('properties', {}))})")
                print(f"    {tool.description}")
            return 0
        result = await client.call_tool(args.tool, arguments)
        if result.is_error:
            text = result.content[0].text if result.content else "the server reported an error"
            print(f"error: {text}", file=sys.stderr)
            return 1
        found = result.structured_content
        if found is None:
            found = [block.text for block in result.content if block.type == "text"]
        print(json.dumps(found, indent=2))
        return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="List a decision server's tools, or call one and print its result.")
    parser.add_argument("target", help="a scenario file or environment.toml (started over stdio), or a server's URL")
    parser.add_argument("tool", nargs="?", help="the tool to call; without it, the tools are listed")
    parser.add_argument("arguments", nargs="?", default="{}", help="the tool's arguments as a JSON object")
    args = parser.parse_args(argv)
    try:
        arguments = json.loads(args.arguments)
    except ValueError:
        arguments = None
    if not isinstance(arguments, dict):
        print(f"error: the arguments must be a JSON object, such as '{{\"product\": \"PX\"}}', not {args.arguments}",
              file=sys.stderr)
        return 2
    refused: list[int] = []
    try:
        return anyio.run(_run, args, arguments, refused)
    except (MCPError, Exception) as error:  # pylint: disable=broad-exception-caught
        if is_url(args.target):
            print(f"error: cannot connect to {args.target}: {refusal(refused) or _cause(error)}", file=sys.stderr)
        else:
            # The server stopped, as when its file does not load; its own message is printed above
            print("error: the decision server stopped; see its message above", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
