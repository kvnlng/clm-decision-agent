"""Tests for decision_agent.py: the reference agent's loop over MCP, with a scripted stand-in for Claude."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# The fake mirrors the Anthropic client's interface, so some arguments go unused.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,too-few-public-methods
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import anyio
import httpx
import pytest
from conftest import free_port, serving
from mcp import Client

from clm_decision import server
from clm_decision.connection import connect
from clm_decision.decision_agent import (MAX_TURNS, NO_TOOLS_SYSTEM, AgentError, ClaudeChat, OllamaChat, answer, main,
                                         server_params)
from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server
from clm_decision.serving import Serve

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
    assert "scenario_sourcing" in offered and "describe" in offered and len(offered) == 18
    assert offered["scenario_sourcing"]["input_schema"]["required"] == ["product"]
    assert "never calculate" in claude.requests[0]["system"].lower()
    assert claude.requests[0]["model"] == "claude-opus-5-5"


@pytest.mark.anyio
async def test_an_invented_figure_is_flagged(mcp_server):
    claude = FakeClaude(message(tool_use("scenario_sourcing", {"product": "PX"}), stop="tool_use"),
                        message(text("Moving everything would earn 812,000 this quarter.")))
    assert (await ask(mcp_server, claude))["unsupported"] == ["812,000"]


@pytest.mark.anyio
async def test_an_invented_citation_is_flagged(mcp_server):
    claude = FakeClaude(message(text("Under Varghese v. China Southern Airlines, 925 F.3d 1339, it fails.")))
    assert (await ask(mcp_server, claude))["citations"] == {"found": [], "not_found": ["925 F.3d 1339"],
                                                           "unchecked": []}


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
async def test_a_citation_echoed_in_a_tool_error_is_not_found(mcp_server):
    claude = FakeClaude(message(tool_use("scenario_sourcing", {"product": "PX",
                                                              "branches": [{"region_b_share": "§ 15.50(b)"}]}),
                                stop="tool_use"),
                        message(text("Under § 15.50(b) the covenant fails.")))
    result = await ask(mcp_server, claude)
    assert "§ 15.50(b)" in result["calls"][0]["error"]
    assert result["citations"]["found"] == [] and result["citations"]["not_found"] == ["§ 15.50(b)"]


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


def test_claude_without_the_agent_extra_says_how_to_install_it(monkeypatch, capsys):
    import sys  # pylint: disable=import-outside-toplevel
    monkeypatch.setitem(sys.modules, "anthropic", None)      # None makes the import fail, as when not installed
    assert main([str(EXAMPLE), "Should we move?"]) == 2
    assert capsys.readouterr().err.startswith("error: Claude needs the agent extra (anthropic is missing)")


def test_a_url_connects_over_http_with_the_key(monkeypatch):
    monkeypatch.setenv("DECISION_SERVER_KEY", "k1")
    port = free_port()
    with serving(server.http_app(build_environment_server(load_environment(EXAMPLE)), "k1"), port):
        result = anyio.run(_ask_over, f"http://127.0.0.1:{port}/mcp")
    assert [call["tool"] for call in result["calls"]] == ["describe"]
    assert result["calls"][0]["result"]["name"] == "Component sourcing"


async def _ask_over(url):
    async with Client(connect(url)) as session:
        return await answer(session, "What is this?", ClaudeChat(FakeClaude(
            message(tool_use("describe", {})), message(text("It is component sourcing."))), "claude-opus-5-5"))


def test_a_refused_key_is_one_line(monkeypatch, capsys):
    monkeypatch.setenv("DECISION_SERVER_KEY", "wrong")
    port = free_port()
    with serving(server.http_app(build_environment_server(load_environment(EXAMPLE)), "k1"), port):
        assert main([f"http://127.0.0.1:{port}/mcp", "Q", "--backend", "ollama"]) == 2
    err = capsys.readouterr().err
    assert "the server refused the call (401), with the key or token in DECISION_SERVER_KEY" in err
    assert "Traceback" not in err


@pytest.mark.anyio
async def test_a_figure_from_the_question_is_not_flagged(mcp_server):
    claude = FakeClaude(message(tool_use("describe", {})),
                        message(text("For the 37 units you mention, see the scenario.")))
    async with Client(mcp_server) as session:
        result = await answer(session, "We could sell 37 units. Should we?", ClaudeChat(claude, "claude-opus-5-5"))
    assert result["unsupported"] == []


def test_the_agent_writes_the_record_beside_the_answer(tmp_path, monkeypatch):
    claude = FakeClaude(message(tool_use("scenario_sourcing", {"product": "PX"})), message(text("Buy half.")))
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda: claude)
    out = tmp_path / "run.answer.json"
    assert main([str(EXAMPLE), "Should we?", "--out", str(out)]) == 0
    record = json.loads((tmp_path / "run.record.json").read_text())
    assert record["question"] == "Should we?" and record["scenarios"][0]["best"] == {"region_b_share": 0.5}
    assert "## The agent's explanation (not checked)" in (tmp_path / "run.record.md").read_text()


def unreachable():
    import anthropic  # pylint: disable=import-outside-toplevel
    return anthropic.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:9/v1/messages"))


def no_credentials():
    # What SDK 2.1.0 raises as it builds the request with no key and no `ant auth login`
    return TypeError('"Could not resolve authentication method. Expected one of api_key, auth_token, or credentials '
                     'to be set. Or for the `Authorization` header to be explicitly omitted"')


class FailingClaude:
    """The Anthropic client when the request cannot be made: it fails as the stream opens."""

    def __init__(self, failure):
        self.failure = failure
        self.messages = self

    def stream(self, **_request):
        return _FailingStream(self.failure)


class _FailingStream:
    def __init__(self, failure):
        self.failure = failure

    async def __aenter__(self):
        raise self.failure()

    async def __aexit__(self, *exc):
        return False


@pytest.mark.parametrize("failure", [unreachable, no_credentials])
@pytest.mark.parametrize("no_tools", [False, True])
def test_a_claude_failure_is_one_line_about_claude(tmp_path, monkeypatch, capsys, no_tools, failure):
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda: FailingClaude(failure))
    target = str(tmp_path / "nowhere.toml") if no_tools else str(EXAMPLE)
    assert main([target, "Should we?", "--out", str(tmp_path / "run.answer.json"),
                 *(["--no-tools"] if no_tools else [])]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: Claude (claude-opus-5-5): ") and "ANTHROPIC_API_KEY" in err
    assert "server stopped" not in err and "Traceback" not in err


def test_no_tools_sends_no_tools_and_its_own_prompt(tmp_path, monkeypatch):
    claude = FakeClaude(message(text("I don't know Meridian's plan.")))
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda: claude)
    out = tmp_path / "run.answer.json"
    assert main([str(tmp_path / "nowhere.toml"), "What is my match?", "--no-tools", "--out", str(out)]) == 0
    assert "tools" not in claude.requests[0] and claude.requests[0]["system"] == NO_TOOLS_SYSTEM
    result = json.loads(out.read_text())
    assert result["calls"] == [] and result["tools"] is False


def test_no_tools_counts_and_records_no_calls_the_model_made_up(tmp_path, monkeypatch, capsys):
    claude = FakeClaude(message(tool_use("describe", {})), message(text("I don't know.")))
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda: claude)
    out = tmp_path / "run.answer.json"
    assert main([str(tmp_path / "nowhere.toml"), "What is my match?", "--no-tools", "--out", str(out)]) == 0
    assert "no tools offered" in capsys.readouterr().err
    assert json.loads((tmp_path / "run.record.json").read_text())["failed_calls"] == []


def test_no_tools_flags_every_figure_not_in_the_question(tmp_path, monkeypatch):
    claude = FakeClaude(message(text("Meridian adds $3,400 a year.")))
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda: claude)
    out = tmp_path / "run.answer.json"
    assert main([str(tmp_path / "nowhere.toml"), "What is my match?", "--no-tools", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["unsupported"] == ["$3,400"]


@pytest.mark.anyio
async def test_no_tools_on_ollama_sends_no_tools():
    ollama = FakeOllama(said("I don't know."))
    async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as http:
        result = await answer(None, QUESTION, OllamaChat(http, "gemma4:26b", system=NO_TOOLS_SYSTEM))
    assert result["answer"] == "I don't know." and result["calls"] == []
    assert ollama.requests[0]["tools"] == [] and ollama.requests[0]["messages"][0]["content"] == NO_TOOLS_SYSTEM


DERIVED_LINE = ("- A rule or fact whose origin is derived was read from the documents by a model and checked "
                "mechanically, not written by the owner: say so, name its source, and say when it is not approved.")


@pytest.mark.anyio
async def test_system_says_when_a_rule_was_derived(mcp_server):
    claude = FakeClaude(message(text("Nothing to add.")))
    await ask(mcp_server, claude)
    ollama = FakeOllama(said("Nothing to add."))
    await ask_ollama(mcp_server, ollama)
    # What each model is sent, not only the constant: the whole line, so a weakened one fails too
    assert DERIVED_LINE in claude.requests[0]["system"].split("\n")
    assert DERIVED_LINE in ollama.requests[0]["messages"][0]["content"].split("\n")


def test_system_is_the_shared_instructions_byte_for_byte():
    import hashlib  # pylint: disable=import-outside-toplevel
    from clm_decision import instructions  # pylint: disable=import-outside-toplevel
    from clm_decision.decision_agent import SYSTEM  # pylint: disable=import-outside-toplevel
    # The text before it moved to instructions.py: 1532 characters with this digest
    assert SYSTEM == instructions.agent_system()
    assert len(SYSTEM) == 1532
    assert hashlib.sha256(SYSTEM.encode()).hexdigest() == \
        "d8c0b5e088f551f09c2fa1d40f2660c082793a18b3dff3c2875641ec7e67c881"


@pytest.mark.anyio
async def test_system_names_only_tools_the_server_offers():
    # The tariff example has no rules tool; without its scenario tools, neither line has a tool to name
    fewer = build_environment_server(load_environment(EXAMPLE), serve=Serve(exclude_tools=("scenario_*",)))
    claude = FakeClaude(message(text("Nothing to add.")))
    await ask(fewer, claude)
    ollama = FakeOllama(said("Nothing to add."))
    await ask_ollama(fewer, ollama)
    for system in (claude.requests[0]["system"], ollama.requests[0]["messages"][0]["content"]):
        assert "never calculate" in system.lower()
        assert "run the scenario tool" not in system and "call rules" not in system


@pytest.mark.anyio
async def test_a_chat_used_again_is_filtered_for_each_server(mcp_server):
    fewer = build_environment_server(load_environment(EXAMPLE), serve=Serve(exclude_tools=("scenario_*",)))
    claude = FakeClaude(message(text("Nothing to add.")), message(text("Nothing to add.")))
    chat = ClaudeChat(claude, "claude-opus-5-5")
    for each in (fewer, mcp_server):
        async with Client(each) as session:
            await answer(session, QUESTION, chat)
    assert "run the scenario tool" not in claude.requests[0]["system"]
    assert "run the scenario tool" in claude.requests[1]["system"]


@pytest.mark.anyio
async def test_the_last_turn_runs_no_tools_the_model_would_never_see(mcp_server):
    claude = FakeClaude(*[message(text("Let me look."), tool_use("describe", {}, f"call_{index}"))
                          for index in range(MAX_TURNS)])
    result = await ask(mcp_server, claude)
    assert len(claude.requests) == MAX_TURNS and len(result["calls"]) == MAX_TURNS - 1
    assert result["stopped"] == "turn limit" and result["answer"] == "Let me look."


@pytest.mark.anyio
async def test_a_tool_call_with_no_tools_goes_back_as_an_error():
    ollama = FakeOllama(said(calls=[call("describe", {})]), said("I don't know."))
    async with httpx.AsyncClient(transport=httpx.MockTransport(ollama)) as http:
        result = await answer(None, QUESTION, OllamaChat(http, "gemma4:26b", system=NO_TOOLS_SYSTEM))
    assert result["calls"] == [{"tool": "describe", "arguments": {}, "result": None, "error": "no tools are offered"}]
    assert result["answer"] == "I don't know."


def test_two_runs_in_the_same_second_keep_both_answers(tmp_path, monkeypatch):
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 4, 12, 0, 0, tzinfo=tz)
    monkeypatch.setattr("clm_decision.decision_agent.datetime", Frozen)
    claude = FakeClaude(message(text("First.")), message(text("Second.")))
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda: claude)
    for _ in range(2):
        assert main([str(tmp_path / "nowhere.toml"), "What is my match?", "--no-tools"]) == 0
    answers = (tmp_path / "runs").glob("*.answer.json")
    assert sorted(json.loads(path.read_text())["answer"] for path in answers) == ["First.", "Second."]
    assert len(list((tmp_path / "runs").glob("*.record.md"))) == 2
