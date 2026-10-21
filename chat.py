"""The test chat: type requests, see the decision server's decisions, and correct them. It talks only MCP."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import anyio
import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from server import KEY_ENV

COMMANDS = "Commands: /routes, /correct <route>, /agreement, /quit"
NEEDS_REVIEW = "That needs review mode: start the chat, or the server it connects to, with --review."


def _answer(name: str, answer: dict) -> str:
    if answer["type"] == "choice":
        return f"{name} {answer['choice']} {answer['confidence']:.2f}"
    return f"{name} {answer[answer['type']]:.2f}"


def format_decision(d: dict) -> list[str]:
    """The decision as indented lines for the terminal; empty parts are left out."""
    others = sorted(((p, name) for name, p in d["probabilities"].items() if name != d["route"]), reverse=True)
    route = f"  route: {d['route']} {d['confidence']:.2f}"
    if others:
        route += f" (next: {others[0][1]} {others[0][0]:.2f})"
    if d.get("source") == "reviewed":
        same = d["clm_route"] == d["route"]
        route += " as you confirmed" if same else f" from your correction; CLM said {d['clm_route']}"
    elif d["uncertain"]:
        route += " uncertain"
    lines = [route]
    if d["answers"]:
        lines.append("  answers: " + "   ".join(_answer(name, a) for name, a in d["answers"].items()))
    for informed in d["informed"]:
        found = " ".join(informed["facts"]) if informed["status"] == "done" else f"failed: {informed['error']}"
        lines.append(f"  facts: {informed['tool']}{':' if informed['status'] == 'done' else ''} {found}")
    lines += [f"  rules: {rule['when']}: {rule['tell']}" for rule in d["rules"]]
    for action in d["actions"]:
        detail = {"done": lambda a: json.dumps(a["result"]), "failed": lambda a: a["error"],
                  "skipped": lambda a: a["reason"]}[action["status"]](action)
        lines.append(f"  actions: {action['tool']} {action['status']}: {detail}")
    return lines


async def _call(session: Client, name: str, arguments: dict, write: Callable) -> dict | None:
    """One MCP tool call; a tool error is written as one line and gives None."""
    result = await session.call_tool(name, arguments)
    if result.is_error:
        write(f"error: {result.content[0].text if result.content else 'the server reported an error'}")
        return None
    return result.structured_content


async def _describe(session: Client, write: Callable) -> None:
    described = await _call(session, "describe", {}, write)
    if described:
        write("Routes:")
        for route in described["routes"]:
            write(f"  {route['name']}: {route['description']}")
        write("Questions:")
        for question in described["questions"]:
            write(f"  {question['name']} ({question['type']}): {question['instructions']}")


async def _command(session: Client, line: str, last: str | None, review: bool, write: Callable) -> None:
    """One /command. `last` is the last request typed, which /correct applies to."""
    command, _, rest = line.partition(" ")
    if command == "/routes":
        await _describe(session, write)
    elif command in ("/correct", "/agreement") and not review:
        write(NEEDS_REVIEW)
    elif command == "/correct":
        if last is None:
            write("Type a request first; /correct applies to the last one.")
        elif await _call(session, "correct", {"text": last, "route": rest.strip()}, write):
            write(f"Saved: the right route for that request is {rest.strip()}.")
    elif command == "/agreement":
        found = await _call(session, "agreement", {}, write)
        if found:
            write(f"CLM's own route agrees with {found['agree']} of your {found['n']} corrections.")
    else:
        write(COMMANDS)


async def run_chat(session: Client, read: Callable[[str], str] = input, write: Callable = print) -> None:
    """Read requests and commands until /quit or the end of input. Review mode is whatever the server offers."""
    review = "correct" in {tool.name for tool in (await session.list_tools()).tools}
    last = None
    while True:
        try:
            line = (await anyio.to_thread.run_sync(read, "you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
        if line == "/quit":
            return
        if line.startswith("/"):
            await _command(session, line, last, review, write)
        elif line:
            last = line
            decided = await _call(session, "decide", {"text": line}, write)
            for out in format_decision(decided) if decided else []:
                write(out)


async def _chat(args) -> int:
    if args.url:
        key = os.environ.get(args.key_env, "").strip()
        if not key:
            print(f"error: set {args.key_env} to the server's key", file=sys.stderr)
            return 2
        async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=300) as http:
            async with Client(streamable_http_client(args.url, http_client=http)) as session:
                await run_chat(session)
        return 0
    server = Path(__file__).with_name("server.py")
    params = StdioServerParameters(command=sys.executable,
                                   args=[str(server), args.scenario] + (["--review"] if args.review else []))
    print("Starting the decision server; loading CLM can take a minute.")
    async with Client(params) as session:
        print(f"Ready. Type a request. {COMMANDS}")
        await run_chat(session)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Try a decision server: type requests and see its decisions.")
    parser.add_argument("scenario", nargs="?", help="start a server for this scenario")
    parser.add_argument("--review", action="store_true", help="allow /correct and /agreement")
    parser.add_argument("--url", help="connect to a running HTTP server instead, such as http://127.0.0.1:8000/mcp")
    parser.add_argument("--key-env", default=KEY_ENV, help="variable holding the HTTP server's key")
    args = parser.parse_args(argv)
    if bool(args.scenario) == bool(args.url):
        parser.error("give a scenario file, or --url for a running server")
    return anyio.run(_chat, args)


if __name__ == "__main__":
    sys.exit(main())
