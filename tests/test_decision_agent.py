"""Tests for decision_agent.py: the reference agent's loop over MCP, with a scripted stand-in for Claude."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# The fake mirrors the Anthropic client's interface, so some arguments go unused.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,too-few-public-methods
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import Client

from decision_agent import answer
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
        return await answer(session, QUESTION, claude, "claude-opus-5-5")


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
async def test_a_cut_off_reply_stops_without_running_its_tools(mcp_server):
    claude = FakeClaude(message(tool_use("scenario_sourcing", {"product": "PX"}), stop="max_tokens"))
    result = await ask(mcp_server, claude)
    assert result["calls"] == [] and result["stopped"] == "max_tokens"
