"""Text generation behind one interface, with an MLX backend and an Ollama backend."""
from __future__ import annotations

import json
import os
import re
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from scenario import Scenario

# A JSON object after <tool_call>, ended by the closing tag, the next call, or the end of the text
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|(?=<tool_call>)|\Z)", re.DOTALL)


# Shown when the model answers with a tool call although no tools were offered
NO_TOOL_REPLY = "I could not complete that request with the information I have."


class GeneratorError(RuntimeError):
    """The generator backend could not produce a reply."""


def _call_id() -> str:
    return f"call_{uuid4().hex[:8]}"


def to_chat_dicts(messages: list[BaseMessage]) -> list[dict]:
    """LangChain messages -> the role/content dicts a tokenizer chat template expects."""
    chat = []
    for message in messages:
        if isinstance(message, HumanMessage):
            chat.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage):
            entry = {"role": "assistant", "content": str(message.content)}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {"type": "function", "function": {"name": call["name"], "arguments": call["args"]}}
                    for call in message.tool_calls
                ]
            chat.append(entry)
        elif isinstance(message, ToolMessage):
            chat.append({"role": "tool", "name": message.name or "", "content": str(message.content)})
        else:
            raise TypeError(f"cannot send a {type(message).__name__} to the generator")
    return chat


def parse_reply(text: str) -> AIMessage:
    """Model output -> AIMessage. Well-formed <tool_call> blocks become tool calls; anything else is text."""
    plain = AIMessage(content=text.strip())
    blocks = _TOOL_CALL.findall(text)
    if not blocks and text.strip().startswith("{") and '"arguments"' in text:
        blocks, text = [text.strip()], "<tool_call>" + text   # the model left the tags out
    if not blocks:
        return plain
    calls = []
    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            return plain
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            return plain
        args = data.get("arguments", {})
        if not isinstance(args, dict):
            return plain
        calls.append({"name": data["name"], "args": args, "id": _call_id()})
    return AIMessage(content=text[: text.index("<tool_call>")].strip(), tool_calls=calls)


def finish_reply(text: str, offered: bool) -> AIMessage:
    """Model output -> the reply for the graph. A tool call nobody offered is never shown to the user."""
    reply = parse_reply(text)
    if offered or not reply.tool_calls:
        return reply
    return AIMessage(content=reply.content or NO_TOOL_REPLY)


class MLXGenerator:
    """Runs in the same process as CLM. `loaded` lets tests pass a (model, tokenizer) pair."""

    def __init__(self, model_id: str, max_tokens: int, loaded: tuple | None = None):
        if loaded is None:
            from mlx_lm import load  # pylint: disable=import-outside-toplevel
            loaded = load(model_id)
        self.model, self.tokenizer = loaded[0], loaded[1]
        self.max_tokens = max_tokens

    def build_prompt(self, system: str, messages: list[BaseMessage], tools: list | None = None) -> str:
        """The full prompt text: system message, conversation and, if given, the tool definitions."""
        chat = [{"role": "system", "content": system}, *to_chat_dicts(messages)]
        schemas = [convert_to_openai_tool(t) for t in tools] if tools else None
        return self.tokenizer.apply_chat_template(chat, tools=schemas, add_generation_prompt=True, tokenize=False)

    def generate(self, system: str, messages: list[BaseMessage], tools: list | None = None) -> AIMessage:
        """One reply; it carries tool calls only when `tools` were offered."""
        from mlx_lm import generate  # pylint: disable=import-outside-toplevel
        text = generate(self.model, self.tokenizer, prompt=self.build_prompt(system, messages, tools),
                        max_tokens=self.max_tokens, verbose=False)
        return finish_reply(text, offered=bool(tools))


class OllamaGenerator:
    """Talks to a local Ollama server through langchain-ollama."""

    def __init__(self, model: str, max_tokens: int, llm=None):
        if llm is None:
            from langchain_ollama import ChatOllama  # pylint: disable=import-outside-toplevel
            # reasoning=False: a thinking model would otherwise spend the token budget before replying
            llm = ChatOllama(model=model, num_predict=max_tokens, reasoning=False)
        self.llm, self.model = llm, model

    def generate(self, system: str, messages: list[BaseMessage], tools: list | None = None) -> AIMessage:
        """One reply; it carries tool calls only when `tools` were offered."""
        import httpx  # pylint: disable=import-outside-toplevel
        import ollama  # pylint: disable=import-outside-toplevel
        llm = self.llm.bind_tools(tools) if tools else self.llm
        try:
            reply = llm.invoke([SystemMessage(content=system), *messages])
        except (httpx.ConnectError, ConnectionError) as error:
            raise GeneratorError("Cannot reach Ollama. Start it with `ollama serve` and try again.") from error
        except ollama.ResponseError as error:
            raise GeneratorError(f"Ollama could not run '{self.model}': {error}. "
                                 f"If the model is missing, run `ollama pull {self.model}`.") from error
        calls = [{"name": call["name"], "args": call["args"], "id": call.get("id") or _call_id()}
                 for call in (reply.tool_calls if tools else [])]
        content = str(reply.content).strip()
        return AIMessage(content=content or ("" if calls else NO_TOOL_REPLY), tool_calls=calls)


class OpenAIGenerator:
    """Talks to any server that offers the OpenAI chat interface (Ollama, vLLM, LM Studio, llama.cpp, ...)."""

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def __init__(self, model: str, max_tokens: int, base_url: str, api_key: str | None = None, llm=None):
        if llm is None:
            from langchain_openai import ChatOpenAI  # pylint: disable=import-outside-toplevel
            # Local servers ignore the key, but the client insists on one
            llm = ChatOpenAI(model=model, max_tokens=max_tokens, base_url=base_url, api_key=api_key or "not-needed",
                             timeout=120, max_retries=1)
        self.llm, self.model, self.base_url = llm, model, base_url

    def generate(self, system: str, messages: list[BaseMessage], tools: list | None = None) -> AIMessage:
        """One reply; it carries tool calls only when `tools` were offered."""
        import openai  # pylint: disable=import-outside-toplevel
        llm = self.llm.bind_tools(tools) if tools else self.llm
        try:
            reply = llm.invoke([SystemMessage(content=system), *messages])
        except openai.APIConnectionError as error:
            raise GeneratorError(f"Cannot reach the model server at {self.base_url}.") from error
        except openai.APIStatusError as error:
            # The server's message is not repeated: it may echo the request, key included
            raise GeneratorError(f"The model server at {self.base_url} refused the request for '{self.model}' "
                                 f"(HTTP {error.status_code}). Check the model name and api_key_env.") from error
        except (ValueError, IndexError, KeyError, TypeError) as error:
            raise GeneratorError(f"The model server at {self.base_url} sent a reply that could not be read "
                                 f"({type(error).__name__}).") from error
        calls = [{"name": call["name"], "args": call["args"], "id": call.get("id") or _call_id()}
                 for call in (reply.tool_calls if tools else [])]
        content = str(reply.content).strip()
        return AIMessage(content=content or ("" if calls else NO_TOOL_REPLY), tool_calls=calls)


def make_generator(scenario: Scenario):
    """The generator backend named in the scenario."""
    if scenario.generator_backend == "openai":
        key = os.environ.get(scenario.generator_key_env or "", "") or None
        return OpenAIGenerator(scenario.generator_model, scenario.max_tokens, scenario.generator_base_url, key)
    if scenario.generator_backend == "ollama":
        return OllamaGenerator(scenario.generator_model, scenario.max_tokens)
    return MLXGenerator(scenario.generator_model, scenario.max_tokens)
