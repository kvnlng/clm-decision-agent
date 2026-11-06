"""The portal's Ask: a model answers a question with a server's tools, every call and result shown.

The model is anything that speaks the OpenAI chat API, Ollama by default, or Claude. The model's key is read from
LLM_API_KEY and never shown. Tools run for real unless a dry run holds them back.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
from mcp import Client, MCPError

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
    return f"{text[:SHOWN]} ... ({len(text) - SHOWN} more characters; tick Full results to see them)"


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
