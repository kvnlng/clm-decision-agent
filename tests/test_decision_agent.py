"""Tests for decision_agent.py: the reference agent's loop over MCP, with a scripted stand-in for Claude."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# The fake mirrors the Anthropic client's interface, so some arguments go unused.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,too-few-public-methods
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from mcp import Client

from decision_agent import AgentError, ClaudeChat, OllamaChat, answer, main, server_params
from environment import load_environment
from environment_server import build_environment_server

EXAMPLE = Path(__file__).parent.parent / "examples" / "tariff" / "environment.toml"
QUESTION = "Should we move Product Line X's sourcing to Region B?"


def text(words):
    return SimpleNamespace(type="text", text=words)


def tool_use(name, arguments, call_id="call_1"):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def message(*blocks, stop="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


class _Stream:
    def __init__(self, reply):
        self.reply = reply

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self.reply


class FakeClaude:
    """Replies with the scripted messages in turn, and keeps each request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = self

    def stream(self, **request):
        self.requests.append({**request, "messages": list(request["messages"])})
        return _Stream(self.replies.pop(0))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def mcp_server():
    return build_environment_server(load_environment(EXAMPLE))


async def ask(mcp_server, claude):
    async with Client(mcp_server) as session:
        return await answer(session, QUESTION, ClaudeChat(claude, "claude-opus-5-5"))


@pytest.mark.anyio
async def test_the_agent_calls_tools_through_mcp(mcp_server):
    claude = FakeClaude(message(tool_use("scenario_sourcing", {"product": "PX"}), stop="tool_use"),
                        message(text("Move half to Region B: the Q3 margin is 704,900.")))
    result = await ask(mcp_server, claude)
    assert result["calls"][0]["tool"] == "scenario_sourcing"
    assert result["calls"][0]["result"]["best"] == {"region_b_share": 0.5}
    assert result["answer"] == "Move half to Region B: the Q3 margin is 704,900." and result["unsupported"] == []
    sent = claude.requests[1]["messages"][-1]["content"][0]
    assert sent["type"] == "tool_result" and sent["tool_use_id"] == "call_1" and "704900" in sent["content"]


@pytest.mark.anyio
async def test_the_tools_offered_are_the_servers(mcp_server):
    claude = FakeClaude(message(text("Nothing to add.")))
    await ask(mcp_server, claude)
    offered = {tool["name"]: tool for tool in claude.requests[0]["tools"]}
    assert "scenario_sourcing" in offered and "describe" in offered and len(offered) == 14
    assert offered["scenario_sourcing"]["input_schema"]["required"] == ["product"]
    assert "never calculate" in claude.requests[0]["system"].lower()
    assert claude.requests[0]["model"] == "claude-opus-5-5"


@pytest.mark.anyio
async def test_an_invented_figure_is_flagged(mcp_server):
    claude = FakeClaude(message(tool_use("scenario_sourcing", {"product": "PX"}), stop="tool_use"),
                        message(text("Moving everything would earn 812,000 this quarter.")))
    assert (await ask(mcp_server, claude))["unsupported"] == ["812,000"]


@pytest.mark.anyio
async def test_a_tool_error_goes_back_to_the_model(mcp_server):
    claude = FakeClaude(message(tool_use("query_suppliers_of", {}), stop="tool_use"),
                        message(text("I could not look that up.")))
    result = await ask(mcp_server, claude)
    sent = claude.requests[1]["messages"][-1]["content"][0]
    assert sent["is_error"] is True and "component" in sent["content"]
    assert result["calls"][0]["error"] and result["answer"] == "I could not look that up."


@pytest.mark.anyio
async def test_a_figure_from_a_tool_error_is_not_flagged(mcp_server):
    claude = FakeClaude(message(tool_use("scenario_sourcing", {"product": "PX", "branches": [{"region_b_share": 0.7}]}),
                                stop="tool_use"),
                        message(text("A share of 0.7 is not among the levers; the allowed shares are 0 and 0.5.")))
    result = await ask(mcp_server, claude)
    assert result["calls"][0]["error"] and "0.7" in result["calls"][0]["error"]
    assert result["unsupported"] == []


@pytest.mark.anyio
@pytest.mark.parametrize("stop", ["max_tokens", "model_context_window_exceeded"])
async def test_a_cut_off_reply_stops_without_running_its_tools(mcp_server, stop):
    claude = FakeClaude(message(tool_use("scenario_sourcing", {"product": "PX"}), stop=stop))
    result = await ask(mcp_server, claude)
    assert result["calls"] == [] and result["stopped"] == stop


class FakeOllama:
    """Ollama's /api/chat, replying with the scripted messages in turn and keeping each request body."""

    def __init__(self, *replies, status=200):
        self.replies, self.status, self.requests = list(replies), status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "model 'gemma4:26b' not found"})
        return httpx.Response(200, json=self.replies.pop(0))


def said(content="", calls=None, done="stop"):
    return {"message": {"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})},
            "done_reason": done}


def call(name, arguments):
    return {"function": {"name": name, "arguments": arguments}}


async def ask_ollama(mcp_server, handler):
    failed = None
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with Client(mcp_server) as session:
            # Caught inside the session, as decision_agent's main does, so MCP's task group does not wrap it
            try:
                return await answer(session, QUESTION, OllamaChat(http, "gemma4:26b"))
            except AgentError as error:
                failed = error
    raise failed


@pytest.mark.anyio
async def test_ollama_tool_calls_round_trip(mcp_server):
    ollama = FakeOllama(said(calls=[call("scenario_sourcing", {"product": "PX"})]),
                        said("Move half to Region B: the Q3 margin is 704,900."))
    result = await ask_ollama(mcp_server, ollama)
    assert result["calls"][0]["result"]["best"] == {"region_b_share": 0.5} and result["unsupported"] == []
    assert result["model"] == "gemma4:26b" and result["answer"].startswith("Move half")
    first, second = ollama.requests  # pylint: disable=unbalanced-tuple-unpacking
    assert first["model"] == "gemma4:26b" and first["stream"] is False
    assert first["messages"][0]["role"] == "system" and "never calculate" in first["messages"][0]["content"].lower()
    offered = {tool["function"]["name"]: tool for tool in first["tools"]}
    assert offered["scenario_sourcing"]["function"]["parameters"]["required"] == ["product"]
    assert second["messages"][-2]["tool_calls"][0]["function"]["name"] == "scenario_sourcing"
    assert second["messages"][-1]["role"] == "tool" and second["messages"][-1]["tool_name"] == "scenario_sourcing"
    assert "704900" in second["messages"][-1]["content"]


@pytest.mark.anyio
async def test_ollama_string_arguments_are_parsed(mcp_server):
    ollama = FakeOllama(said(calls=[call("scenario_sourcing", '{"product": "PX"}')]), said("Done."))
    result = await ask_ollama(mcp_server, ollama)
    assert result["calls"][0]["arguments"] == {"product": "PX"} and result["calls"][0]["error"] is None


@pytest.mark.anyio
async def test_ollama_bad_arguments_and_unknown_tools_go_back_as_errors(mcp_server):
    ollama = FakeOllama(said(calls=[call("scenario_sourcing", "{not json"), call("forecast_q4", {})]), said("Sorry."))
    result = await ask_ollama(mcp_server, ollama)
    bad, unknown = result["calls"]
    assert "not valid JSON" in bad["error"] and unknown["error"]
    sent = [message for message in ollama.requests[1]["messages"] if message["role"] == "tool"]
    assert len(sent) == 2 and "not valid JSON" in sent[0]["content"]


@pytest.mark.anyio
async def test_ollama_cut_off_stops(mcp_server):
    result = await ask_ollama(mcp_server, FakeOllama(said(calls=[call("describe", {})], done="length")))
    assert result["calls"] == [] and result["stopped"] == "max_tokens"


@pytest.mark.anyio
async def test_ollama_unreachable_is_one_line(mcp_server):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)
    with pytest.raises(AgentError, match=r"^cannot reach Ollama at http://127\.0\.0\.1:11434"):
        await ask_ollama(mcp_server, refuse)
    with pytest.raises(AgentError, match=r"^Ollama has no model gemma4:26b"):
        await ask_ollama(mcp_server, FakeOllama(status=404))


def test_the_server_gets_the_callers_environment(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "from-the-shell")
    params = server_params("examples/tariff/environment.toml")
    assert (params.env or {}).get("NEO4J_PASSWORD") == "from-the-shell"
    assert params.args[-1] == "examples/tariff/environment.toml"


def test_a_server_that_stops_is_one_line(tmp_path, capsys):
    missing = tmp_path / "nowhere.toml"
    assert main([str(missing), "a question", "--backend", "ollama", "--ollama-url", "http://127.0.0.1:9"]) == 2
    err = capsys.readouterr().err
    assert "error: the decision server stopped" in err and "Traceback" not in err
