"""Tests for server.py: the scenario's decisions over MCP, in process and over HTTP."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import socket
import threading
import time

import httpx2
import pytest
import uvicorn
import fake_tools
from conftest import BASE_TOML, FakeRouter, decision
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

import clm_router
import server
from decision import decide, routed_text
from record import load_records
from scenario import load_scenario
from tools import load_tools

TEXT = "my app crashes"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def parts(write_scenario):
    """-> (scenario, router, tools) for BASE_TOML, with CLM always answering Tech."""
    scenario = load_scenario(write_scenario())
    return scenario, FakeRouter(decision("Tech", answers={"urgent": {"type": "noul", "noul": 0.2}})), \
        load_tools(scenario)


async def tool_names(mcp_server):
    async with Client(mcp_server) as client:
        return sorted(tool.name for tool in (await client.list_tools()).tools)


@pytest.mark.anyio
async def test_lists_tools_without_review(parts):
    assert await tool_names(server.build_server(*parts, review=False)) == \
        ["check_call", "decide", "decide_many", "describe"]


@pytest.mark.anyio
async def test_lists_review_tools(parts):
    assert await tool_names(server.build_server(*parts, review=True)) == \
        ["agreement", "check_call", "correct", "decide", "decide_many", "describe"]


@pytest.mark.anyio
async def test_only_deciding_can_change_anything(parts):
    async with Client(server.build_server(*parts, review=True)) as client:
        hints = {tool.name: tool.annotations.read_only_hint for tool in (await client.list_tools()).tools}
    assert hints == {"decide": False, "decide_many": False, "check_call": True, "describe": True, "correct": False,
                     "agreement": True}


@pytest.mark.anyio
async def test_decide_over_mcp(parts):
    scenario, router, tools = parts
    async with Client(server.build_server(scenario, router, tools, review=False)) as client:
        result = await client.call_tool("decide", {"text": TEXT})
    assert not result.is_error
    assert result.structured_content == decide(TEXT, scenario, FakeRouter(router.decisions), tools)


@pytest.mark.anyio
async def test_decide_over_mcp_accepts_context(write_scenario):
    scenario = load_scenario(write_scenario())
    urgent = {"urgent": {"type": "noul", "noul": 0.9}}
    router = FakeRouter(decision("Billing", answers=urgent), asks=[urgent])
    fake_tools.CALLS.clear()
    async with Client(server.build_server(scenario, router, load_tools(scenario), review=False)) as client:
        result = await client.call_tool("decide", {"text": TEXT, "context": {"booking": "PNR1"}})
    assert not result.is_error
    assert fake_tools.CALLS[0][1]["context"] == {"booking": "PNR1"}


@pytest.mark.anyio
async def test_correct_saves_and_is_reused(parts):
    scenario, router, tools = parts
    router.asks = [{"urgent": {"type": "noul", "noul": 0.2}}]     # Billing's informing tool has urgent asked again
    async with Client(server.build_server(scenario, router, tools, review=True)) as client:
        saved = await client.call_tool("correct", {"text": TEXT, "route": "Billing"})
        again = await client.call_tool("decide", {"text": TEXT})
        agreement = await client.call_tool("agreement", {})
    assert saved.structured_content == {"saved": True}
    records, _ = load_records(scenario.record_path)
    assert [(r["state_text"], r["human"], r["clm"]["route"]) for r in records] == [(TEXT, "Billing", "Tech")]
    assert (again.structured_content["route"], again.structured_content["source"]) == ("Billing", "reviewed")
    # Agreement replays CLM without the reuse of corrections, so it shows CLM's own disagreement
    assert agreement.structured_content == {"n": 1, "agree": 0}


@pytest.mark.anyio
async def test_correct_unknown_route_errors(parts):
    scenario, router, tools = parts
    async with Client(server.build_server(scenario, router, tools, review=True)) as client:
        result = await client.call_tool("correct", {"text": TEXT, "route": "Shipping"})
    assert result.is_error
    assert "unknown route 'Shipping'" in result.content[0].text
    assert not scenario.record_path.exists()


@pytest.mark.anyio
async def test_check_call_over_mcp(parts):
    async with Client(server.build_server(*parts, review=False)) as client:
        result = await client.call_tool("check_call", {"text": TEXT, "call": {"name": "restart", "args": {}}})
    assert result.structured_content == {"p": 0.9, "approved": True}


@pytest.mark.anyio
async def test_describe_over_mcp(parts):
    async with Client(server.build_server(*parts, review=False)) as client:
        result = await client.call_tool("describe", {})
    assert [route["name"] for route in result.structured_content["routes"]] == ["Billing", "Tech"]


@pytest.fixture
def http_url(parts):
    """A server on a free port, behind the key "k1"; -> its MCP address."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = server.http_app(server.build_server(*parts, review=False), "k1")
    runner = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    while not runner.started:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}/mcp"
    runner.should_exit = True
    thread.join()


@pytest.mark.anyio
async def test_http_with_key_decides(http_url):
    async with httpx2.AsyncClient(headers={"Authorization": "Bearer k1"}) as http:
        async with Client(streamable_http_client(http_url, http_client=http)) as client:
            result = await client.call_tool("decide", {"text": TEXT})
    assert result.structured_content["route"] == "Tech"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer k2"}, {"Authorization": "k1"}])
def test_http_without_the_right_key_is_401(http_url, headers):
    response = httpx2.post(http_url, headers=headers, json={})
    assert (response.status_code, response.json()) == (401, {"error": "missing or wrong key"})


def test_http_without_key_env_refuses_to_start(write_scenario, monkeypatch, capsys):
    monkeypatch.delenv("DECISION_SERVER_KEY", raising=False)
    assert server.main([str(write_scenario()), "--http"]) == 2
    assert "set DECISION_SERVER_KEY to the key callers must send" in capsys.readouterr().err


def test_a_broken_scenario_is_one_line(write_scenario, capsys):
    assert server.main([str(write_scenario(BASE_TOML.replace('tool = "act"', 'tool = "x"')))]) == 2
    assert capsys.readouterr().err.strip().startswith("error: actions[0].tool: unknown tool 'x'")


def test_the_server_logs_only_warnings(parts):
    # Every request a tool makes would otherwise be logged to the chat's terminal
    assert server.build_server(*parts, review=False).settings.log_level == "WARNING"


@pytest.mark.anyio
async def test_a_slow_tool_does_not_hold_up_other_calls(write_scenario):
    import anyio  # pylint: disable=import-outside-toplevel
    text = (BASE_TOML.replace('tool = "act"', 'tool = "slow"').replace('when = "urgent >= 0.5"\n', "")
            .replace('routes = ["Billing"]', 'routes = ["Tech"]'))
    scenario = load_scenario(write_scenario(text))
    router = FakeRouter(decision("Tech", answers={"urgent": {"type": "noul", "noul": 0.2}}))
    fake_tools.GATE.clear()
    async with Client(server.build_server(scenario, router, load_tools(scenario), review=False)) as client:
        async with anyio.create_task_group() as group:
            group.start_soon(client.call_tool, "decide", {"text": TEXT})
            await anyio.sleep(0.2)                       # decide is now waiting inside the slow tool
            started = time.monotonic()
            checked = await client.call_tool("check_call", {"text": TEXT, "call": {"name": "x"}})
            waited = time.monotonic() - started
            fake_tools.GATE.set()
    assert checked.structured_content == {"p": 0.9, "approved": True}
    assert waited < 1


def test_clm_that_cannot_load_is_one_line(write_scenario, monkeypatch, capsys):
    def no_cuda(*args):
        raise clm_router.ClmError("CLM was set to run on cuda, which this machine does not have (available: cpu)")
    monkeypatch.setattr(clm_router, "load_engine", no_cuda)
    assert server.main([str(write_scenario())]) == 2
    assert capsys.readouterr().err.strip().splitlines()[-1] == \
        "error: CLM was set to run on cuda, which this machine does not have (available: cpu)"


@pytest.mark.anyio
async def test_correct_saves_the_text_the_route_saw(write_scenario):
    scenario = load_scenario(write_scenario(BASE_TOML.replace(
        "confidence_threshold = 0.6", 'confidence_threshold = 0.6\ninform = ["lookup"]')))
    seen = routed_text(TEXT, ["The queue is A."])
    router = FakeRouter({TEXT: decision("Tech"), seen: decision("Tech")})
    async with Client(server.build_server(scenario, router, load_tools(scenario), review=True)) as client:
        saved = await client.call_tool("correct", {"text": TEXT, "route": "Billing", "context": {"booking": "B"}})
    assert not saved.is_error
    records, _ = load_records(scenario.record_path)
    assert [r["state_text"] for r in records] == [seen]
    assert router.verdicts == {seen: "Billing"}


@pytest.mark.anyio
async def test_decide_many_over_mcp(parts):
    scenario, router, tools = parts
    async with Client(server.build_server(scenario, router, tools, review=False)) as client:
        result = await client.call_tool("decide_many", {"requests": [{"text": TEXT}, {"text": "other"}]})
        empty = await client.call_tool("decide_many", {"requests": []})
        too_many = await client.call_tool("decide_many", {"requests": [{"text": "x"}] * 1001})
        no_text = await client.call_tool("decide_many", {"requests": [{"context": {}}]})
    assert [d["route"] for d in result.structured_content["decisions"]] == ["Tech", "Tech"]
    assert empty.structured_content == {"decisions": []}
    assert too_many.is_error and "at most 1000" in too_many.content[0].text
    assert no_text.is_error and "requests[0]" in no_text.content[0].text


@pytest.mark.anyio
async def test_decide_many_accepts_a_null_context_as_decide_does(parts):
    async with Client(server.build_server(*parts, review=False)) as client:
        result = await client.call_tool("decide_many", {"requests": [{"text": TEXT, "context": None}]})
    assert not result.is_error


OPTIONS_TOML = BASE_TOML.replace('inform = ["lookup"]\n', '''inform = ["lookup", "alternatives"]

[routes.Billing.options]
tool = "alternatives"
instructions = "Which refund suits them?"
''')
REFUNDS = [{"id": "card", "text": "Back to the card"}, {"id": "credit", "text": "Store credit"}]


@pytest.fixture
def offered(write_scenario):
    """A review server whose Billing route ranks two refunds; -> (scenario, server)."""
    fake_tools.OPTIONS["value"] = REFUNDS
    scenario = load_scenario(write_scenario(OPTIONS_TOML))
    yield scenario, server.build_server(scenario, FakeRouter(decision("Tech")), load_tools(scenario), review=True)
    fake_tools.OPTIONS["value"] = []


@pytest.mark.anyio
async def test_correct_saves_answers_and_the_option(offered):
    scenario, mcp = offered
    async with Client(mcp) as client:
        saved = await client.call_tool("correct", {"text": TEXT, "route": "Billing", "answers": {"urgent": True},
                                                   "option": "credit", "context": {"booking": "B"}})
    assert saved.structured_content == {"saved": True}
    records, _ = load_records(scenario.record_path)
    entry = records[0]
    assert (entry["answers"], entry["option"], entry["options"]) == ({"urgent": True}, "credit", REFUNDS)
    # Saved by tool, so training gives each question only the facts it hears when deciding
    assert entry["facts"] == {"lookup": ["The queue is A."]}


@pytest.mark.anyio
@pytest.mark.parametrize("answers, problem", [({"mood": True}, "unknown question 'mood'"),
                                               ({"urgent": "yes"}, "questions.urgent: needs true or false")])
async def test_correct_with_a_bad_answer_saves_nothing(offered, answers, problem):
    scenario, mcp = offered
    async with Client(mcp) as client:
        result = await client.call_tool("correct", {"text": TEXT, "route": "Billing", "answers": answers})
    assert result.is_error and problem in result.content[0].text
    assert not scenario.record_path.exists()


@pytest.mark.anyio
async def test_correct_with_an_option_not_offered_saves_nothing(offered):
    scenario, mcp = offered
    async with Client(mcp) as client:
        result = await client.call_tool("correct", {"text": TEXT, "route": "Billing", "option": "voucher"})
        on_tech = await client.call_tool("correct", {"text": TEXT, "route": "Tech", "option": "card"})
    assert result.is_error and "'voucher' was not among the options" in result.content[0].text
    assert on_tech.is_error and "routes.Tech has no options" in on_tech.content[0].text
    assert not scenario.record_path.exists()
