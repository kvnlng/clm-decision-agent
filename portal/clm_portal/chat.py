"""The portal's command-line chat with a running MCP server, through any model.

    clm-portal chat URL [--model M] [--model-url U] [--backend openai|claude] [--dry-run [TOOL]] [--full]

It connects to a running server, offers a model the server's tools, and prints every call and result, so you can see
where an answer came from and judge it yourself. Tools run for real unless --dry-run holds them back.

The model is anything that speaks the OpenAI chat API, Ollama by default, or Claude with --backend claude. The
server's key or token is read from DECISION_SERVER_KEY, and the model's from LLM_API_KEY; neither is printed.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass

import anyio
import anyio.to_thread
import httpx
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client

KEY_ENV = "DECISION_SERVER_KEY"
LLM_KEY_ENV = "LLM_API_KEY"
STARTER_MODEL = "granite4:micro"
OLLAMA_V1 = "http://127.0.0.1:11434/v1"
CLAUDE_MODEL = "claude-opus-5-5"
MAX_TURNS = 20
SHOWN = 1500
ALL = "*"
SYSTEM = "Answer using the tools you are given. If they cannot answer, say so plainly rather than guessing."
NOT_SENT = ("Dry run: this call was recorded but not carried out, so it has no result. Do not invent one. Say that "
            "the call was not carried out.")


class ClientError(RuntimeError):
    """Something the user must fix; the message is one line saying what."""


@dataclass(frozen=True)
class Call:
    """A tool call the model asked for: arguments that are not a JSON object become `invalid`."""
    id: str
    name: str
    arguments: dict
    invalid: str | None = None


@dataclass(frozen=True)
class Turn:
    """One reply: its text, the calls it asks for, why it was cut off (or None), and the message for the history."""
    text: str
    calls: list[Call]
    stop: str | None
    message: dict


@dataclass(frozen=True)
class DryRun:
    """Which tools are held back: none, some by name, or all (ALL)."""
    tools: frozenset | set

    def holds(self, name: str) -> bool:
        """Whether a call to this tool is shown but not sent."""
        return ALL in self.tools or name in self.tools


class OpenAIModel:
    """Anything that speaks the OpenAI chat API, such as Ollama, LM Studio, vLLM or a hosted model."""

    def __init__(self, http: httpx.AsyncClient, model: str, url: str, key: str | None = None):
        self.http, self.model, self.url, self.key = http, model, url.rstrip("/"), key

    def describe(self) -> str:
        """The model and where it is, for the banner."""
        return f"{self.model} at {self.url}"

    def start(self) -> list[dict]:
        """The history before any question."""
        return [{"role": "system", "content": SYSTEM}]

    def tools(self, listed) -> list[dict]:
        """The server's tools as function definitions."""
        return [{"type": "function", "function": {"name": tool.name, "description": tool.description or "",
                                                  "parameters": tool.input_schema}} for tool in listed]

    async def turn(self, messages: list[dict], tools: list[dict]) -> Turn:
        """One reply."""
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else None
        try:
            response = await self.http.post(f"{self.url}/chat/completions", headers=headers, timeout=600, json={
                "model": self.model, "messages": messages, "tools": tools, "stream": False})
        except httpx.HTTPError as error:
            raise ClientError(f"cannot reach the model at {self.url}: {error or type(error).__name__}; "
                              "for the starter model, is Ollama running?") from error
        if response.status_code == 404:
            raise ClientError(f"no model {self.model} at {self.url}; with Ollama, run `ollama pull {self.model}`")
        if response.status_code != 200:
            raise ClientError(f"the model at {self.url} answered {response.status_code}: {response.text[:200]}")
        try:
            choice = response.json()["choices"][0]
            message = choice.get("message") or {}
            raw = message.get("tool_calls") or []
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            raise ClientError(f"the model at {self.url} sent a reply that is not a chat completion: "
                              f"{response.text[:200]}") from error
        kept = {"role": "assistant", "content": message.get("content") or ""}
        if raw:
            kept["tool_calls"] = raw
        stop = "the reply was too long" if choice.get("finish_reason") == "length" else None
        return Turn((message.get("content") or "").strip(), [_call(item) for item in raw], stop, kept)

    def results(self, pairs: list[tuple[Call, str]]) -> list[dict]:
        """One tool message per call."""
        return [{"role": "tool", "tool_call_id": call.id, "content": text} for call, text in pairs]


def _call(item: dict) -> Call:
    function = item.get("function") or {}
    name, arguments = function.get("name", ""), function.get("arguments", "{}")
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return Call(item.get("id", ""), name, {}, invalid=str(arguments))
    if not isinstance(parsed, dict):
        return Call(item.get("id", ""), name, {}, invalid=str(arguments))
    return Call(item.get("id", ""), name, parsed)


class ClaudeModel:
    """Claude through the Anthropic SDK: streamed, with adaptive thinking."""

    def __init__(self, client, model: str = CLAUDE_MODEL):
        self.client, self.model = client, model

    def describe(self) -> str:
        """The model, for the banner."""
        return f"{self.model} through the Anthropic API"

    def start(self) -> list[dict]:
        """The history before any question; the instructions go in `system`."""
        return []

    def tools(self, listed) -> list[dict]:
        """The server's tools as Anthropic tool definitions."""
        return [{"name": tool.name, "description": tool.description or "", "input_schema": tool.input_schema}
                for tool in listed]

    async def turn(self, messages: list[dict], tools: list[dict]) -> Turn:
        """One streamed reply; ClientError when the API fails, as with a wrong or missing key or a rate limit."""
        try:
            async with self.client.messages.stream(model=self.model, max_tokens=16000, system=SYSTEM, tools=tools,
                                                   thinking={"type": "adaptive"}, messages=messages) as stream:
                reply = await stream.get_final_message()
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise ClientError(f"the Anthropic API failed: {error or type(error).__name__}") from error
        text = "\n".join(block.text for block in reply.content if block.type == "text").strip()
        calls = [Call(block.id, block.name, dict(block.input or {})) for block in reply.content
                 if block.type == "tool_use"]
        stop = f"the reply stopped: {reply.stop_reason}" if reply.stop_reason in (
            "max_tokens", "model_context_window_exceeded", "refusal") else None
        return Turn(text, calls, stop, {"role": "assistant", "content": reply.content})

    def results(self, pairs: list[tuple[Call, str]]) -> list[dict]:
        """One user message of tool_result blocks."""
        return [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id, "content": text}
                                             for call, text in pairs]}]


def shown(text: str, full: bool) -> str:
    """A result for the screen, shortened unless the user asked for all of it."""
    if full or len(text) <= SHOWN:
        return text
    return f"{text[:SHOWN]} ... ({len(text) - SHOWN} more characters; --full shows them)"


async def _send(client: Client, call: Call) -> tuple[str, bool]:
    """Carry out one call -> (its result as text, whether it failed)."""
    try:
        result = await client.call_tool(call.name, call.arguments)
    except MCPError as error:
        return f"Error: {error}", True
    text = "\n".join(block.text for block in result.content if block.type == "text")
    if result.is_error:
        return text or "Error: the server reported an error", True
    if result.structured_content is not None:
        return json.dumps(result.structured_content, indent=2), False
    return text, False


async def _handle(client: Client, call: Call, dry_run: DryRun, show, full: bool) -> str:
    """Show one call, carry it out unless it is held back -> the text for the model."""
    arguments = call.invalid if call.invalid is not None else json.dumps(call.arguments)
    if dry_run.holds(call.name):
        show(f"  [dry run, not sent: {call.name} {arguments}]")
        return NOT_SENT
    show(f"  [tool: {call.name} {arguments}]")
    if call.invalid is not None:
        show("  [not sent: the arguments are not valid JSON]")
        return "Error: the arguments were not valid JSON; send a JSON object."
    text, failed = await _send(client, call)
    show(f"  [{'error' if failed else 'result'}: {shown(text, full)}]")
    return text


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
async def answer(client: Client, model, tools: list[dict], messages: list[dict], question: str, dry_run: DryRun,
                 show=print, full: bool = False) -> None:
    """Add a question to the conversation and let the model work until it answers, printing every call. A reply
    cut off runs none of its calls. When the model fails or does not finish, the question is taken back out of the
    conversation, so the next one starts from a history the model will accept."""
    mark = len(messages)
    messages.append({"role": "user", "content": question})
    for _ in range(MAX_TURNS):
        try:
            turn = await model.turn(messages, tools)
        except ClientError as error:
            show(f"  [error: {error}]")
            del messages[mark:]
            return
        if turn.stop:
            show(f"  [cut off: {turn.stop}; none of its tool calls were run]")
            del messages[mark:]
            return
        messages.append(turn.message)
        if not turn.calls:
            show(turn.text or "(the model gave no answer)")
            return
        if turn.text:
            show(turn.text)
        pairs = [(call, await _handle(client, call, dry_run, show, full)) for call in turn.calls]
        messages.extend(model.results(pairs))
    show(f"  [stopped: no answer after {MAX_TURNS} turns]")
    del messages[mark:]


def check_dry_run(names: set, tools: list[str]) -> None:
    """Refuse a name the server does not have: a mistyped tool would otherwise run for real."""
    unknown = sorted(set(names) - set(tools) - {ALL})
    if unknown:
        raise ClientError(f"--dry-run names {', '.join(unknown)}, which the server does not have; "
                          f"its tools are {', '.join(sorted(tools))}")


def banner(url: str, model: str, dry_run: DryRun) -> str:
    """What the user must know before typing anything."""
    if ALL in dry_run.tools:
        tools = "Dry run: every tool call is shown and not sent."
    elif dry_run.tools:
        tools = f"Dry run for {', '.join(sorted(dry_run.tools))}, not sent. Every other tool will run for real."
    else:
        tools = "Tools will run for real, including any that act. --dry-run holds them back."
    return f"Server: {url}\nModel: {model}\n{tools}\nType a question; /tools lists the tools; /quit leaves."


@contextlib.asynccontextmanager
async def _connect(url: str, refused: list[int]):
    key = os.environ.get(KEY_ENV)
    headers = {"Authorization": f"Bearer {key}"} if key else None
    async with httpx.AsyncClient(headers=headers, follow_redirects=True,
                                 timeout=httpx.Timeout(30, read=600)) as http:
        async def note(response) -> None:
            if response.status_code in (401, 403):
                refused.append(response.status_code)
        http.event_hooks["response"].append(note)
        async with streamable_http_client(url, http_client=http) as streams:
            yield streams


def _list(listed) -> str:
    return "\n".join(f"  {tool.name}: {tool.description or ''}".rstrip() for tool in listed)


async def _chat(args, http: httpx.AsyncClient, refused: list[int]) -> int:
    model = _model(args, http)
    dry_run = DryRun(frozenset(args.dry_run or ()))
    async with Client(_connect(args.url, refused)) as client:
        listed = (await client.list_tools()).tools
        check_dry_run(set(dry_run.tools), [tool.name for tool in listed])
        print(banner(args.url, model.describe(), dry_run))
        print(_list(listed))
        tools, messages = model.tools(listed), model.start()
        while True:
            try:
                question = (await anyio.to_thread.run_sync(input, "\nyou> ", abandon_on_cancel=True)).strip()
            except EOFError:
                return 0
            if question == "/quit":
                return 0
            if question == "/tools":
                print(_list(listed))
            elif question:
                await answer(client, model, tools, messages, question, dry_run, print, args.full)


def _model(args, http: httpx.AsyncClient):
    if args.backend == "openai":
        return OpenAIModel(http, args.model or STARTER_MODEL, args.model_url or OLLAMA_V1,
                           os.environ.get(LLM_KEY_ENV))
    try:
        from anthropic import AsyncAnthropic  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise ClientError("--backend claude needs the anthropic package, which the portal installs with "
                          "clm-decision-agent[agent]") from error
    return ClaudeModel(AsyncAnthropic(), args.model or CLAUDE_MODEL)


def _cause(error: BaseException) -> str:
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return str(error) or type(error).__name__


def _user_error(error: BaseException) -> ClientError | None:
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return error if isinstance(error, ClientError) else None


async def _run(args, refused: list[int]) -> int:
    async with httpx.AsyncClient() as http:
        return await _chat(args, http, refused)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The chat's arguments, as `clm-portal chat` takes them."""
    parser.add_argument("url", help="the server's URL, such as http://127.0.0.1:8000/mcp")
    parser.add_argument("--backend", choices=("openai", "claude"), default="openai",
                        help="openai: anything that speaks the OpenAI chat API (default); claude: the Anthropic API")
    parser.add_argument("--model", help=f"default: {STARTER_MODEL} with openai, {CLAUDE_MODEL} with claude")
    parser.add_argument("--model-url", help=f"the model's OpenAI-style address (default: Ollama, {OLLAMA_V1})")
    parser.add_argument("--dry-run", nargs="?", const=ALL, action="append", metavar="TOOL",
                        help="show this tool's calls without sending them; with no name, every tool's; repeatable")
    parser.add_argument("--full", action="store_true", help="print results in full")


def run(args: argparse.Namespace) -> int:
    """Chat until the user leaves; returns the process exit code."""
    refused: list[int] = []
    try:
        return anyio.run(_run, args, refused)
    except KeyboardInterrupt:
        return 0
    except (Exception, BaseExceptionGroup) as error:  # pylint: disable=broad-exception-caught
        known = _user_error(error)
        if known:
            print(f"error: {known}", file=sys.stderr)
        elif refused:
            print(f"error: {args.url} refused the call ({refused[0]}); set the key or token in {KEY_ENV}",
                  file=sys.stderr)
        else:
            print(f"error: cannot connect to {args.url}: {_cause(error)}", file=sys.stderr)
        return 2
