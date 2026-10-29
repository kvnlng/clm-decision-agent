"""The reference agent: Claude answers a question from an environment's tools, over MCP, and its figures are checked.

    uv run --group demo python decision_agent.py ENVIRONMENT "QUESTION" [--model claude-opus-5-5] [--out FILE]

It starts server.py for the environment, offers Claude the server's tools and nothing else, and lets it call them
until it answers. Every figure in the answer is then checked against what the tools returned (number_check.py), and
the question, the answer, every call and the check are written as JSON, by default to the environment's runs/
folder. Credentials are the Anthropic SDK's usual ones (ANTHROPIC_API_KEY or `ant auth login`); none are stored.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters

from number_check import unsupported

MODEL = "claude-opus-5-5"
MAX_TURNS = 20
SYSTEM = """You answer an executive's question about their organisation using only the tools you are given. They \
belong to a decision server whose owner wrote every query, rule, calculation and lever in it.

- Start with describe, to see what the environment offers.
- Never calculate, estimate or convert a figure yourself. Every number in your answer must appear in a tool result.
- To compare alternatives, run the scenario tool: it carries every number from query to rule to result, and it \
checks every limit on every branch. Do not combine queries and decisions yourself to reach a figure.
- Say which branch is best and why, which branches were rejected and by which rule, with its reason, and any \
dependency behind a rejection, naming it.
- Cite what you relied on: the tool, and for a scenario the branch and the step (query, decision or calculation).
- If no tool can answer part of the question, say so plainly rather than guessing."""


def _tools(listed) -> list[dict]:
    """The server's tools as Anthropic tool definitions, with the server's own schemas."""
    return [{"name": tool.name, "description": tool.description or "", "input_schema": tool.input_schema,
             "eager_input_streaming": True} for tool in listed]


async def _run(session: Client, name: str, arguments: dict) -> dict:
    """One MCP call -> {"tool", "arguments", "result", "error"}."""
    result = await session.call_tool(name, arguments)
    if result.is_error:
        message = result.content[0].text if result.content else "the server reported an error"
        return {"tool": name, "arguments": arguments, "result": None, "error": message}
    return {"tool": name, "arguments": arguments, "result": result.structured_content, "error": None}


async def _tool_results(session: Client, reply, calls: list[dict]) -> list[dict]:
    """Run each tool the reply asks for, adding to `calls` -> the tool_result blocks to send back."""
    results = []
    for use in (block for block in reply.content if block.type == "tool_use"):
        call = await _run(session, use.name, dict(use.input or {}))
        calls.append(call)
        content = call["error"] if call["error"] else json.dumps(call["result"])
        results.append({"type": "tool_result", "tool_use_id": use.id, "content": content,
                        **({"is_error": True} if call["error"] else {})})
    return results


async def answer(session: Client, question: str, client, model: str = MODEL) -> dict:
    """Ask Claude the question with the session's tools -> {"question", "answer", "calls", "unsupported", "stopped"}.
    A reply cut off (max_tokens) or refused stops the loop without running the tools it asked for."""
    tools = _tools((await session.list_tools()).tools)
    messages: list[dict] = [{"role": "user", "content": question}]
    calls: list[dict] = []
    stopped = None
    reply = None
    for _ in range(MAX_TURNS):
        async with client.messages.stream(model=model, max_tokens=16000, system=SYSTEM, tools=tools,
                                          thinking={"type": "adaptive"}, messages=messages) as stream:
            reply = await stream.get_final_message()
        messages.append({"role": "assistant", "content": reply.content})
        if reply.stop_reason in ("max_tokens", "refusal"):
            stopped = reply.stop_reason
            break
        if reply.stop_reason == "pause_turn":
            continue
        results = await _tool_results(session, reply, calls)
        if not results:
            break
        messages.append({"role": "user", "content": results})
    else:
        stopped = "turn limit"
    text = "\n".join(block.text for block in (reply.content if reply else []) if block.type == "text").strip()
    return {"question": question, "answer": text, "calls": calls,
            "unsupported": unsupported(text, [call["result"] for call in calls if call["result"] is not None]),
            "stopped": stopped}


async def _main(args) -> int:
    import anthropic  # pylint: disable=import-outside-toplevel
    server = Path(__file__).with_name("server.py")
    params = StdioServerParameters(command=sys.executable, args=[str(server), args.environment])
    async with Client(params) as session:
        result = await answer(session, args.question, anthropic.AsyncAnthropic(), args.model)
    print(result["answer"] or "(no answer)")
    if result["stopped"]:
        print(f"\nStopped early: {result['stopped']}.")
    if result["unsupported"]:
        print(f"\nFigures no tool returned: {', '.join(result['unsupported'])}")
    out = Path(args.out) if args.out else (Path(args.environment).resolve().parent / "runs" /
                                           f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.answer.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**result, "model": args.model}, indent=2), encoding="utf-8")
    print(f"\n{len(result['calls'])} tool calls; written to {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Answer a question from an environment's tools with Claude.")
    parser.add_argument("environment", help="path to the environment's environment.toml")
    parser.add_argument("question")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", help="where to write the result as JSON (default: the environment's runs/ folder)")
    args = parser.parse_args(argv)
    return anyio.run(_main, args)


if __name__ == "__main__":
    sys.exit(main())
