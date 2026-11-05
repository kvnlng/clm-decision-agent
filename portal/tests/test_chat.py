"""Tests for the portal's chat, against a stand-in MCP server and a stand-in model; neither needs a network or a key."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json
from types import SimpleNamespace

import httpx
import pytest
from mcp import Client
from mcp.server import MCPServer

import clm_portal
from clm_portal import chat as tc


class Tracker:
    """A stand-in server with one tool that acts and counts its calls."""

    def __init__(self):
        self.tickets = 0
        self.server = MCPServer("stand-in")

        @self.server.tool()
        def open_ticket(summary: str) -> dict:
            """Open a ticket."""
            self.tickets += 1
            return {"number": self.tickets, "summary": summary}

        @self.server.tool()
        def lookup(name: str) -> dict:
            """Look something up."""
            return {"found": name}


def reply(content="", calls=(), finish="stop"):
    tool_calls = [{"id": f"call_{index}", "type": "function",
                   "function": {"name": name, "arguments": arguments if isinstance(arguments, str)
                                else json.dumps(arguments)}}
                  for index, (name, arguments) in enumerate(calls)]
    return {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content,
                                                              "tool_calls": tool_calls or None}}]}


def scripted(replies, seen=None):
    """A model that gives these replies in turn, keeping each request in `seen`."""
    replies = iter(replies)

    def answer(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        item = next(replies)
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)
    return tc.OpenAIModel(httpx.AsyncClient(transport=httpx.MockTransport(answer)), "small", "http://model/v1")


async def ask(tracker, model, question="Open a ticket", dry_run=frozenset(), history=None):
    lines: list[str] = []
    history = model.start() if history is None else history
    async with Client(tracker.server) as client:
        tools = model.tools((await client.list_tools()).tools)
        await tc.answer(client, model, tools, history, question, tc.DryRun(dry_run), lines.append)
    return "\n".join(lines)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def tracker():
    return Tracker()


@pytest.mark.anyio
async def test_by_default_a_tool_really_runs_and_its_result_is_shown(tracker):
    model = scripted([reply(calls=[("open_ticket", {"summary": "Printer"})]), reply("Ticket 1 is open.")])
    shown = await ask(tracker, model)
    assert tracker.tickets == 1
    assert 'open_ticket {"summary": "Printer"}' in shown and '"number": 1' in shown
    assert "Ticket 1 is open." in shown


@pytest.mark.anyio
async def test_the_result_goes_back_to_the_model(tracker):
    seen: list[dict] = []
    await ask(tracker, scripted([reply(calls=[("lookup", {"name": "x"})]), reply("Done")], seen))
    back = seen[1]["messages"][-1]
    assert back["role"] == "tool" and back["tool_call_id"] == "call_0" and json.loads(back["content"]) == {"found": "x"}


@pytest.mark.anyio
async def test_a_dry_run_shows_the_call_does_not_send_it_and_tells_the_model(tracker):
    seen: list[dict] = []
    model = scripted([reply(calls=[("open_ticket", {"summary": "Printer"})]), reply("It was not opened.")], seen)
    shown = await ask(tracker, model, dry_run={tc.ALL})
    assert tracker.tickets == 0
    assert 'dry run, not sent: open_ticket {"summary": "Printer"}' in shown
    assert "not carried out" in seen[1]["messages"][-1]["content"]


@pytest.mark.anyio
async def test_a_dry_run_of_one_tool_leaves_the_others_running(tracker):
    model = scripted([reply(calls=[("lookup", {"name": "x"}), ("open_ticket", {"summary": "y"})]), reply("Done")])
    shown = await ask(tracker, model, dry_run={"open_ticket"})
    assert tracker.tickets == 0 and '"found": "x"' in shown and "dry run, not sent: open_ticket" in shown


def test_a_dry_run_of_a_tool_the_server_does_not_have_is_refused_before_any_call():
    with pytest.raises(tc.ClientError, match="open_tikcet.*lookup, open_ticket"):
        tc.check_dry_run({"open_tikcet"}, ["lookup", "open_ticket"])


@pytest.mark.anyio
async def test_a_follow_up_keeps_the_conversation(tracker):
    seen: list[dict] = []
    model = scripted([reply("First"), reply("Second")], seen)
    history = model.start()
    await ask(tracker, model, "One", history=history)
    await ask(tracker, model, "Two", history=history)
    assert [item["content"] for item in seen[1]["messages"] if item["role"] == "user"] == ["One", "Two"]


@pytest.mark.anyio
async def test_a_tool_error_is_shown_and_given_to_the_model(tracker):
    seen: list[dict] = []
    shown = await ask(tracker, scripted([reply(calls=[("no_such_tool", {})]), reply("Sorry")], seen))
    assert "[error: Unknown tool" in shown and "Unknown tool" in seen[1]["messages"][-1]["content"]


@pytest.mark.anyio
async def test_arguments_that_are_not_json_are_returned_to_the_model_unsent(tracker):
    seen: list[dict] = []
    await ask(tracker, scripted([reply(calls=[("open_ticket", "{not json")]), reply("Sorry")], seen))
    assert tracker.tickets == 0 and "JSON" in seen[1]["messages"][-1]["content"]


@pytest.mark.anyio
async def test_a_reply_cut_off_runs_nothing_and_leaves_the_conversation_usable(tracker):
    model = scripted([reply(calls=[("open_ticket", {"summary": "y"})], finish="length")])
    history = model.start()
    shown = await ask(tracker, model, history=history)
    assert tracker.tickets == 0 and "cut off" in shown and history == model.start()


@pytest.mark.anyio
async def test_a_model_that_never_stops_calling_tools_is_stopped(tracker):
    model = scripted([reply(calls=[("lookup", {"name": "x"})])] * tc.MAX_TURNS)
    history = model.start()
    shown = await ask(tracker, model, history=history)
    assert f"{tc.MAX_TURNS} turns" in shown and history == model.start()


@pytest.mark.anyio
async def test_a_missing_model_says_how_to_get_it_and_the_chat_goes_on(tracker):
    model = scripted([httpx.Response(404, json={"error": "model not found"})])
    history = model.start()
    assert "ollama pull small" in await ask(tracker, model, history=history)
    assert history == model.start()


def test_a_long_result_is_shortened_unless_full_is_asked_for():
    assert "more characters; --full" in tc.shown("x" * 5000, full=False)
    assert tc.shown("x" * 5000, full=True) == "x" * 5000


def test_the_banner_names_the_model_and_whether_tools_run():
    text = tc.banner("http://server/mcp", "small at http://model/v1", tc.DryRun(frozenset()))
    assert "testing only" not in text.lower() and "small at http://model/v1" in text and "run for real" in text
    assert "not sent" in tc.banner("http://server/mcp", "small", tc.DryRun({tc.ALL}))
    assert "open_ticket" in tc.banner("http://server/mcp", "small", tc.DryRun({"open_ticket"}))


def test_the_key_is_never_in_the_banner(monkeypatch):
    monkeypatch.setenv("DECISION_SERVER_KEY", "secret-value")
    assert "secret-value" not in tc.banner("http://server/mcp", "small", tc.DryRun(frozenset()))


def test_chat_subcommand_takes_the_old_arguments():
    args = clm_portal.parse(["chat", "http://server/mcp", "--backend", "claude", "--model", "m", "--model-url",
                             "http://model/v1", "--dry-run", "open_ticket", "--dry-run", "--full"])
    assert args.command == "chat" and args.url == "http://server/mcp" and args.backend == "claude"
    assert args.model == "m" and args.model_url == "http://model/v1" and args.full
    assert args.dry_run == ["open_ticket", tc.ALL]


class FakeStream:
    """A finished stream holding one reply."""

    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get_final_message(self):
        return self.message


class FakeAnthropic:
    """Just enough of the Anthropic client: replies in turn, keeping each request's messages."""

    def __init__(self, replies):
        self.replies, self.seen = iter(replies), []
        self.messages = self

    def stream(self, **request):
        self.seen.append(list(request["messages"]))
        return FakeStream(next(self.replies))


def claude_reply(stop, *blocks):
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


@pytest.mark.anyio
async def test_claude_calls_a_tool_and_gets_its_result(tracker):
    fake = FakeAnthropic([
        claude_reply("tool_use", SimpleNamespace(type="tool_use", id="t1", name="open_ticket",
                                                 input={"summary": "Printer"})),
        claude_reply("end_turn", SimpleNamespace(type="text", text="Opened."))])
    shown = await ask(tracker, tc.ClaudeModel(fake))
    assert tracker.tickets == 1 and "Opened." in shown
    assert fake.seen[1][-1]["content"][0]["tool_use_id"] == "t1"


@pytest.mark.anyio
async def test_claude_cut_off_runs_nothing(tracker):
    fake = FakeAnthropic([claude_reply("max_tokens", SimpleNamespace(type="tool_use", id="t1", name="open_ticket",
                                                                     input={"summary": "y"}))])
    assert "cut off" in await ask(tracker, tc.ClaudeModel(fake)) and tracker.tickets == 0


class FailingAnthropic:
    """An Anthropic client whose every reply fails, as a wrong key or a rate limit does."""

    def __init__(self, error):
        self.error, self.messages = error, self

    def stream(self, **_):
        raise self.error


@pytest.mark.anyio
async def test_a_claude_failure_is_shown_and_the_question_taken_back_out(tracker):
    history: list[dict] = []
    shown = await ask(tracker, tc.ClaudeModel(FailingAnthropic(Exception("invalid x-api-key"))), history=history)
    assert "[error: the Anthropic API failed: invalid x-api-key]" in shown and not history


@pytest.mark.anyio
async def test_a_reply_that_is_not_a_chat_completion_is_shown_and_the_question_taken_back_out(tracker):
    for body in ("<html>sign in</html>", json.dumps({"object": "error"}), json.dumps({"choices": []}), "[1]"):
        history = [{"role": "system", "content": "x"}]
        shown = await ask(tracker, scripted([httpx.Response(200, text=body)]), history=history)
        assert "is not a chat completion" in shown and len(history) == 1, body


class FailingTools:
    """A client whose every call fails with these content blocks."""

    def __init__(self, *blocks):
        self.blocks = list(blocks)

    async def call_tool(self, *_):
        return SimpleNamespace(is_error=True, content=self.blocks, structured_content=None)


@pytest.mark.anyio
async def test_a_tool_error_is_every_text_block_of_it_whatever_comes_first():
    image = SimpleNamespace(type="image", data="", mime_type="image/png")
    one, two = SimpleNamespace(type="text", text="one"), SimpleNamespace(type="text", text="two")
    call, send = tc.Call("c1", "lookup", {}), tc._send  # pylint: disable=protected-access
    assert await send(FailingTools(image, one, two), call) == ("one\ntwo", True)
    assert await send(FailingTools(image), call) == ("Error: the server reported an error", True)
