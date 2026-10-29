"""The reference agent: a model answers a question from an environment's tools, over MCP, and its figures are checked.

    uv run --group demo python decision_agent.py ENVIRONMENT "QUESTION" [--backend claude|ollama] [--model M]
                                                 [--out FILE]

It starts server.py for the environment, offers the model the server's tools and nothing else, and lets it call them
until it answers. The model is Claude through the Anthropic API (the default, claude-opus-5-5) or a local model
through Ollama (gemma4:26b by default); both get the same instructions and tools. Every figure in the answer is then
checked against what the tools returned (number_check.py), and the question, the answer, every call and the check
are written as JSON, by default to the environment's runs/ folder. Claude's credentials are the Anthropic SDK's
usual ones (ANTHROPIC_API_KEY or `ant auth login`); none are stored.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import anyio
import httpx
from mcp import Client, StdioServerParameters

from number_check import unsupported

MODEL = "claude-opus-5-5"
LOCAL_MODEL = "gemma4:26b"
OLLAMA_URL = "http://127.0.0.1:11434"
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


class AgentError(RuntimeError):
    """The model cannot be reached or used; the message is one line saying what to do."""


@dataclass(frozen=True)
class Use:
    """A tool call the model asked for: arguments it sent that are not a JSON object become `invalid`."""
    id: str
    name: str
    input: dict
    invalid: str | None = None


@dataclass(frozen=True)
class Turn:
    """One reply: its text, the tools it asks for, why it stopped early (or None), whether to ask again
    (a paused turn), and the model's own message for the history."""
    text: str
    uses: list[Use]
    stop: str | None
    again: bool
    message: dict


class Chat(Protocol):
    """A model the loop can talk to; each keeps its own message format."""
    model: str

    def tools(self, listed) -> list[dict]:
        """The server's tools in this model's format."""

    def start(self, question: str) -> list[dict]:
        """The first messages."""

    async def turn(self, messages: list[dict], tools: list[dict]) -> Turn:
        """The model's next reply."""

    def results(self, pairs: list[tuple[Use, dict]]) -> list[dict]:
        """The messages that carry the calls' results back."""


def _content(call: dict) -> str:
    return call["error"] if call["error"] else json.dumps(call["result"])


class ClaudeChat:
    """Claude through the Anthropic SDK: streamed, with adaptive thinking."""

    def __init__(self, client, model: str = MODEL):
        self.client, self.model = client, model

    def tools(self, listed) -> list[dict]:
        """The server's tools as Anthropic tool definitions, with the server's own schemas."""
        return [{"name": tool.name, "description": tool.description or "", "input_schema": tool.input_schema,
                 "eager_input_streaming": True} for tool in listed]

    def start(self, question: str) -> list[dict]:
        """The question; the instructions go in `system`."""
        return [{"role": "user", "content": question}]

    async def turn(self, messages: list[dict], tools: list[dict]) -> Turn:
        """One streamed reply."""
        async with self.client.messages.stream(model=self.model, max_tokens=16000, system=SYSTEM, tools=tools,
                                               thinking={"type": "adaptive"}, messages=messages) as stream:
            reply = await stream.get_final_message()
        text = "\n".join(block.text for block in reply.content if block.type == "text").strip()
        uses = [Use(block.id, block.name, dict(block.input or {})) for block in reply.content
                if block.type == "tool_use"]
        stop = reply.stop_reason if reply.stop_reason in ("max_tokens", "model_context_window_exceeded",
                                                          "refusal") else None
        return Turn(text, uses, stop, reply.stop_reason == "pause_turn",
                    {"role": "assistant", "content": reply.content})

    def results(self, pairs: list[tuple[Use, dict]]) -> list[dict]:
        """One user message of tool_result blocks."""
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": use.id, "content": _content(call),
             **({"is_error": True} if call["error"] else {})} for use, call in pairs]}]


class OllamaChat:
    """A local model through Ollama's /api/chat."""

    def __init__(self, http, model: str = LOCAL_MODEL, url: str = OLLAMA_URL):
        self.http, self.model, self.url = http, model, url.rstrip("/")

    def tools(self, listed) -> list[dict]:
        """The server's tools as Ollama function definitions."""
        return [{"type": "function", "function": {"name": tool.name, "description": tool.description or "",
                                                  "parameters": tool.input_schema}} for tool in listed]

    def start(self, question: str) -> list[dict]:
        """The instructions as a system message, then the question."""
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]

    async def turn(self, messages: list[dict], tools: list[dict]) -> Turn:
        """One reply; raises AgentError when Ollama or the model is not there."""
        try:
            response = await self.http.post(f"{self.url}/api/chat", timeout=600, json={
                "model": self.model, "messages": messages, "tools": tools, "stream": False})
        except httpx.HTTPError as error:
            raise AgentError(f"cannot reach Ollama at {self.url}: {error}; is `ollama serve` running?") from error
        if response.status_code == 404:
            raise AgentError(f"Ollama has no model {self.model}: run `ollama pull {self.model}`")
        if response.status_code != 200:
            raise AgentError(f"Ollama at {self.url} answered {response.status_code}: {response.text[:200]}")
        data = response.json()
        message = data.get("message") or {}
        uses = [_ollama_use(index, item.get("function") or {}) for index, item in
                enumerate(message.get("tool_calls") or [])]
        stop = "max_tokens" if data.get("done_reason") == "length" else None
        return Turn((message.get("content") or "").strip(), uses, stop, False, message)

    def results(self, pairs: list[tuple[Use, dict]]) -> list[dict]:
        """One tool message per call."""
        return [{"role": "tool", "tool_name": use.name, "content": _content(call)} for use, call in pairs]


def _ollama_use(index: int, function: dict) -> Use:
    """A tool call from Ollama, whose arguments may come as a JSON object or as a string holding one."""
    name, arguments = function.get("name", ""), function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return Use(f"call_{index}", name, {}, invalid=arguments)
    if not isinstance(arguments, dict):
        return Use(f"call_{index}", name, {}, invalid=json.dumps(arguments))
    return Use(f"call_{index}", name, arguments)


async def _run(session: Client, use: Use) -> dict:
    """One MCP call -> {"tool", "arguments", "result", "error"}."""
    if use.invalid is not None:
        return {"tool": use.name, "arguments": use.invalid, "result": None,
                "error": "the arguments were not valid JSON for a tool call; send an object"}
    result = await session.call_tool(use.name, use.input)
    if result.is_error:
        message = result.content[0].text if result.content else "the server reported an error"
        return {"tool": use.name, "arguments": use.input, "result": None, "error": message}
    return {"tool": use.name, "arguments": use.input, "result": result.structured_content, "error": None}


async def answer(session: Client, question: str, chat: Chat) -> dict:
    """Ask the model the question with the session's tools -> {"question", "answer", "calls", "unsupported",
    "stopped", "model"}. A reply cut off (max_tokens, or the context window full) or refused stops the loop without
    running the tools it asked for."""
    tools = chat.tools((await session.list_tools()).tools)
    messages = chat.start(question)
    calls: list[dict] = []
    stopped, text = None, ""
    for _ in range(MAX_TURNS):
        turn = await chat.turn(messages, tools)
        messages.append(turn.message)
        text = turn.text
        if turn.stop:
            stopped = turn.stop
            break
        if turn.again:
            continue
        if not turn.uses:
            break
        pairs = [(use, await _run(session, use)) for use in turn.uses]
        calls += [call for _, call in pairs]
        messages += chat.results(pairs)
    else:
        stopped = "turn limit"
    # A tool's error is tool output too: a figure quoted from it is one the model was given
    returned = [call["error"] if call["error"] else call["result"] for call in calls]
    return {"question": question, "answer": text, "calls": calls, "unsupported": unsupported(text, returned),
            "stopped": stopped, "model": chat.model}


async def _main(args) -> int:
    server = Path(__file__).with_name("server.py")
    params = StdioServerParameters(command=sys.executable, args=[str(server), args.environment])
    async with contextlib.AsyncExitStack() as stack:
        if args.backend == "ollama":
            chat = OllamaChat(await stack.enter_async_context(httpx.AsyncClient()), args.model or LOCAL_MODEL,
                              args.ollama_url)
        else:
            import anthropic  # pylint: disable=import-outside-toplevel
            chat = ClaudeChat(anthropic.AsyncAnthropic(), args.model or MODEL)
        session = await stack.enter_async_context(Client(params))
        started = time.perf_counter()
        try:
            result = await answer(session, args.question, chat)
        except AgentError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        result["seconds"] = round(time.perf_counter() - started, 1)
    print(result["answer"] or "(no answer)")
    if result["stopped"]:
        print(f"\nStopped early: {result['stopped']}.")
    if result["unsupported"]:
        print(f"\nFigures no tool returned: {', '.join(result['unsupported'])}")
    out = Path(args.out) if args.out else (Path(args.environment).resolve().parent / "runs" /
                                           f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.answer.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{len(result['calls'])} tool calls in {result['seconds']} s; written to {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Answer a question from an environment's tools with Claude or a "
                                                 "local model.")
    parser.add_argument("environment", help="path to the environment's environment.toml")
    parser.add_argument("question")
    parser.add_argument("--backend", choices=("claude", "ollama"), default="claude")
    parser.add_argument("--model", help=f"default: {MODEL} with claude, {LOCAL_MODEL} with ollama")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--out", help="where to write the result as JSON (default: the environment's runs/ folder)")
    args = parser.parse_args(argv)
    return anyio.run(_main, args)


if __name__ == "__main__":
    sys.exit(main())
