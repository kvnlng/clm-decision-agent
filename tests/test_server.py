"""Tests for server.py: the scenario's decisions over MCP, in process and over HTTP."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json
import time

import httpx2
import pytest
import fake_tools
from conftest import BASE_TOML, FakeRouter, decision, free_port, serving
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from clm_decision import clm_router
from clm_decision import server
from clm_decision import tracing
from clm_decision.decision import decide, routed_text
from clm_decision.record import append_record, load_records, new_key
from clm_decision.scenario import ScenarioError, load_scenario
from clm_decision.serving import Serve
from clm_decision.tools import load_tools

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
    port = free_port()
    with serving(server.http_app(server.build_server(*parts, review=False), "k1"), port):
        yield f"http://127.0.0.1:{port}/mcp"


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


@pytest.mark.parametrize("scheme", ["bearer", "BEARER", "Bearer"])
def test_http_takes_the_bearer_scheme_in_any_case(http_url, scheme):
    response = httpx2.post(http_url, headers={"Authorization": f"{scheme} k1"}, json={})
    assert response.status_code != 401


def test_http_refusing_a_key_names_the_scheme(http_url):
    response = httpx2.post(http_url, json={})
    assert response.status_code == 401 and response.headers["www-authenticate"].startswith("Bearer")


INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                         "clientInfo": {"name": "probe", "version": "0"}}}


@pytest.fixture
def allowed_url(parts):
    """A server on a free port, behind the key "k1", accepting the host decisions.example.org; -> its address."""
    port = free_port()
    app = server.http_app(server.build_server(*parts, review=False), "k1", allowed_hosts=["decisions.example.org"])
    with serving(app, port):
        yield f"http://127.0.0.1:{port}/mcp"


@pytest.mark.parametrize("headers, status", [
    ({"Host": "decisions.example.org"}, 200),                   # as an HTTPS client sends it, with no port
    ({"Host": "decisions.example.org:443"}, 200),
    ({"Host": "decisions.example.org", "Origin": "https://decisions.example.org"}, 200),
    ({}, 200),                                                   # 127.0.0.1:PORT, as on the server's own computer
    ({"Host": "elsewhere.example.org"}, 421),
    ({"Host": "decisions.example.org", "Origin": "https://elsewhere.example.org"}, 403)])
def test_an_allowed_host_is_taken_with_or_without_a_port(allowed_url, headers, status):
    response = httpx2.post(allowed_url, json=INITIALIZE,
                           headers={"Authorization": "Bearer k1", "Accept": "application/json, text/event-stream",
                                    **headers})
    assert response.status_code == status


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
@pytest.mark.parametrize("context", [[], "", 0, False])
async def test_decide_many_refuses_a_context_that_is_not_an_object(parts, context):
    async with Client(server.build_server(*parts, review=False)) as client:
        result = await client.call_tool("decide_many", {"requests": [{"text": TEXT, "context": context}]})
    assert result.is_error and "requests[0]" in result.content[0].text


@pytest.mark.anyio
async def test_a_text_over_the_limit_is_refused_before_clm(parts):
    scenario, _, tools = parts
    long = "x" * (server.MAX_TEXT + 1)
    calls = {"decide": {"text": long}, "decide_many": {"requests": [{"text": TEXT}, {"text": long}]},
             "check_call": {"text": long, "call": {"tool": "refund"}}, "correct": {"text": long, "route": "Billing"}}
    router = FakeRouter(decision("Tech"))
    async with Client(server.build_server(scenario, router, tools, review=True)) as client:
        results = {tool: await client.call_tool(tool, arguments) for tool, arguments in calls.items()}
        at_limit = await client.call_tool("decide", {"text": "x" * server.MAX_TEXT})
        long_call = await client.call_tool("check_call", {"text": TEXT, "call": {"tool": "refund", "note": long}})
    for tool, result in results.items():
        assert result.is_error and f"at most {server.MAX_TEXT} characters" in result.content[0].text, tool
    assert "requests[1].text" in results["decide_many"].content[0].text
    assert long_call.is_error and "call: at most" in long_call.content[0].text
    assert not at_limit.is_error and not scenario.record_path.exists()


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


@pytest.mark.anyio
async def test_tools_offers_only_those_and_describe(parts):
    chosen = server.build_server(*parts, review=True, serve=Serve(tools=("decide*",), exclude_tools=("decide_many",)))
    assert await tool_names(chosen) == ["decide", "describe"]


@pytest.mark.anyio
async def test_describe_says_which_tools_were_left_out(parts):
    async with Client(server.build_server(*parts, review=False, serve=Serve(exclude_tools=("check_call",)))) as client:
        described = (await client.call_tool("describe", {})).structured_content
    assert described["left_out"] == ["check_call"] and "read_only" not in described


@pytest.mark.anyio
async def test_by_default_describe_says_nothing_of_selection_or_read_only(parts):
    async with Client(server.build_server(*parts, review=False)) as client:
        described = (await client.call_tool("describe", {})).structured_content
    assert "left_out" not in described and "read_only" not in described


@pytest.mark.anyio
async def test_a_read_only_server_decides_without_acting(write_scenario):
    scenario = load_scenario(write_scenario())
    urgent = {"urgent": {"type": "noul", "noul": 0.9}}
    router = FakeRouter(decision("Billing", answers=urgent), asks=[urgent, urgent])
    fake_tools.CALLS.clear()
    async with Client(server.build_server(scenario, router, load_tools(scenario), review=False,
                                          serve=Serve(read_only=True))) as client:
        made = (await client.call_tool("decide", {"text": TEXT})).structured_content
        many = (await client.call_tool("decide_many", {"requests": [{"text": TEXT}]})).structured_content
        described = (await client.call_tool("describe", {})).structured_content
        hints = {tool.name: tool.annotations.read_only_hint for tool in (await client.list_tools()).tools}
        said = client.instructions
    assert made["actions"] == [{"tool": "act", "status": "skipped", "reason": "read-only server"}]
    assert many["decisions"][0]["actions"] == made["actions"]
    assert "act" not in [name for name, _ in fake_tools.CALLS]
    assert described["read_only"] is True and all(hints.values())
    assert "read-only" in said and "may run the scenario's acting tools" not in said


@pytest.mark.parametrize("argv", [["--read-only", "--review"], ["--review"]])
def test_read_only_with_review_stops_with_one_line(write_scenario, argv, capsys, monkeypatch):
    monkeypatch.setattr(server, "load", lambda path: pytest.fail("CLM was loaded"))
    text = BASE_TOML + ("\n[serve]\nread_only = true\n" if argv == ["--review"] else "")
    assert server.main([str(write_scenario(text)), *argv]) == 2
    assert capsys.readouterr().err.strip() == \
        "error: a read-only server cannot offer review mode: correct writes the record"


@pytest.mark.anyio
async def test_the_instructions_name_only_the_deciding_tools_offered(parts):
    async with Client(server.build_server(*parts, review=False)) as client:
        plain = client.instructions
    async with Client(server.build_server(*parts, review=False, serve=Serve(exclude_tools=("decide",)))) as client:
        many = client.instructions
    async with Client(server.build_server(*parts, review=False, serve=Serve(tools=("check_call",)))) as client:
        neither = client.instructions
    assert plain.endswith("then decide with the text of a request. decide may run the scenario's acting tools.")
    assert many.endswith("then decide_many with the texts of requests. decide_many may run the scenario's acting "
                         "tools.")
    assert "decide" not in neither and neither.endswith("Call describe to see the routes and questions.")


def test_admit_derived_with_a_scenario_stops_with_one_line(write_scenario, capsys, monkeypatch):
    monkeypatch.setattr(server, "load", lambda path: pytest.fail("CLM was loaded"))
    assert server.main([str(write_scenario()), "--admit-derived", "proposed"]) == 2
    assert capsys.readouterr().err.strip() == \
        "error: --admit-derived is for environments; a CLM scenario has no derived rules"


def test_a_review_tool_may_be_excluded_without_review(write_scenario, monkeypatch):
    monkeypatch.setattr(server, "load", lambda path: ("scenario", "router", "tools"))
    monkeypatch.setattr(server, "build_server", lambda *args, **kwargs: type("Ran", (), {"run": lambda *a: None})())
    assert server.main([str(write_scenario(BASE_TOML + '\n[serve]\nexclude_tools = ["correct"]\n'))]) == 0


@pytest.mark.anyio
async def test_a_server_without_review_accepts_excluding_a_review_tool(parts):
    excluding = server.build_server(*parts, review=False, serve=Serve(exclude_tools=("correct", "agreement")))
    assert await tool_names(excluding) == ["check_call", "decide", "decide_many", "describe"]


def test_an_unknown_tool_stops_before_clm_loads(write_scenario, capsys, monkeypatch):
    monkeypatch.setattr(server, "load", lambda path: pytest.fail("CLM was loaded"))
    assert server.main([str(write_scenario()), "--tools", "decide,nope"]) == 2
    assert capsys.readouterr().err.strip() == "error: tools: 'nope' matches no tool; the tools are: decide, " \
        "decide_many, check_call, describe, correct, agreement"


def test_a_bad_serve_table_stops_with_one_line(write_scenario, capsys):
    assert server.main([str(write_scenario(BASE_TOML + "\n[serve]\ntools = \"decide\"\n"))]) == 2
    assert capsys.readouterr().err.strip().startswith("error: serve.tools: a list of tool names")


def test_the_command_line_replaces_the_files_selection(write_scenario, monkeypatch):
    built = {}

    class Ran:  # pylint: disable=too-few-public-methods
        """A server that only notes how it was run."""

        def run(self, transport):
            built["transport"] = transport

    def build(*args, **kwargs):  # pylint: disable=unused-argument
        built.update(kwargs)
        return Ran()
    monkeypatch.setattr(server, "load", lambda path: ("scenario", "router", "tools"))
    monkeypatch.setattr(server, "build_server", build)
    text = BASE_TOML + '\n[serve]\ntools = ["decide"]\nexclude_tools = ["check_call"]\n'
    assert server.main([str(write_scenario(text)), "--tools", "decide_many", "--read-only"]) == 0
    assert built["serve"] == Serve(("decide_many",), ("check_call",), True) and built["transport"] == "stdio"


def test_review_mode_stops_before_clm_loads_on_a_record_under_another_key(write_scenario, capsys, monkeypatch):
    monkeypatch.setattr(server, "load", lambda path: pytest.fail("CLM was loaded"))
    scenario = load_scenario(write_scenario())
    append_record(scenario.record_path, {"kind": "route", "state_text": "User: a", "human": "Tech",
                                         "clm": {"route": "Billing", "confidence": 0.8}}, key=new_key())
    assert server.main([str(scenario.path), "--review"]) == 2
    assert "1 line of scenario.corrections.jsonl is encrypted and the scenario names no key" in capsys.readouterr().err


def test_loading_says_how_many_record_lines_it_could_not_read(write_scenario, capsys, monkeypatch):
    class Router(FakeRouter):
        """A router that reuses none of the record's corrections."""

        def load_verdicts(self, records, scenario):  # pylint: disable=unused-argument
            return 0
    monkeypatch.setattr(clm_router, "make_router", lambda scenario: (Router(decision("Tech")), "test heads"))
    scenario = load_scenario(write_scenario())
    append_record(scenario.record_path, {"kind": "route", "state_text": "User: a", "human": "Tech",
                                         "clm": {"route": "Billing", "confidence": 0.8}}, key=new_key())
    server.load(str(scenario.path))
    assert "1 unreadable line skipped in scenario.corrections.jsonl" in capsys.readouterr().err


KEY_A, KEY_B = "agent-a-key-0123456789", "agent-b-key-9876543210"


def key_file(tmp_path, *lines):
    path = tmp_path / "keys.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_either_key_opens(parts, tmp_path):
    keys = server.KeySet.from_sources(None, key_file(tmp_path, KEY_A, KEY_B))
    port = free_port()
    with serving(server.http_app(server.build_server(*parts, review=False), keys), port):
        url = f"http://127.0.0.1:{port}/mcp"
        status = [httpx2.post(url, headers={"Authorization": f"Bearer {key}"}, json={}).status_code
                  for key in (KEY_A, KEY_B, "wrong-key-0123456789")]
    assert status[0] != 401 and status[1] != 401 and status[2] == 401


def test_the_variable_and_the_file_both_count(tmp_path):
    keys = server.KeySet.from_sources("short", key_file(tmp_path, KEY_A, KEY_A))
    assert len(keys.ids()) == 2
    assert keys.match(b"short") and keys.match(KEY_A.encode()) and keys.match(b"other") is None


def test_short_file_key_refused(tmp_path):
    path = key_file(tmp_path, KEY_A, "too-short")
    with pytest.raises(ScenarioError, match=r"keys.txt line 2: a key must be at least 16") as error:
        server.KeySet.from_sources(None, path)
    assert "too-short" not in str(error.value)


@pytest.mark.parametrize("file_exists, message", [(False, "cannot read the key file"), (True, "no keys")])
def test_no_keys_or_no_file_refused(tmp_path, file_exists, message):
    path = key_file(tmp_path, "# nothing here", "") if file_exists else tmp_path / "missing.txt"
    with pytest.raises(ScenarioError, match=message):
        server.KeySet.from_sources(None, path)


def test_comment_lines_ignored(tmp_path):
    keys = server.KeySet.from_sources(None, key_file(tmp_path, "# agent a", "", f"  {KEY_A}  ", "#" + KEY_B))
    assert keys.match(KEY_A.encode()) and keys.match(KEY_B.encode()) is None and len(keys.ids()) == 1


def test_last_used_updates(tmp_path):
    keys = server.KeySet.from_sources(None, key_file(tmp_path, KEY_A, KEY_B))
    assert not keys.last_used
    before = time.time()
    key_id = keys.match(KEY_B.encode())
    assert keys.last_used.keys() == {key_id} and keys.last_used[key_id] >= before
    assert keys.match(b"wrong") is None and keys.last_used.keys() == {key_id}


@pytest.mark.anyio
async def test_key_id_is_not_the_key_and_reaches_the_trace(parts, tmp_path):
    keys = server.KeySet.from_sources(None, key_file(tmp_path, KEY_A, KEY_B))
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    port = free_port()
    app = server.http_app(server.build_server(*parts, review=False, trace=writer), keys)
    with serving(app, port):
        async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {KEY_B}"}) as http:
            async with Client(streamable_http_client(f"http://127.0.0.1:{port}/mcp", http_client=http)) as client:
                await client.call_tool("decide", {"text": TEXT})
    writer.close()
    trace = (tmp_path / "trace.jsonl").read_text()
    records = [json.loads(line) for line in trace.splitlines()]
    assert [record["caller"] for record in records if record["tool"] == "decide"] == [{"key_id": keys.ids()[1]}]
    for key in (KEY_A, KEY_B):
        assert key not in trace and key not in "".join(keys.ids()) and key[:8] not in "".join(keys.ids())


def test_key_flags_are_checked(write_scenario, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DECISION_SERVER_KEY", raising=False)
    assert server.main([str(write_scenario()), "--http", "--auth", "oauth", "--key-file", "k.txt"]) == 2
    assert "--key-file is for --auth key" in capsys.readouterr().err
    assert server.main([str(write_scenario()), "--http", "--key-file", str(key_file(tmp_path, "too-short"))]) == 2
    err = capsys.readouterr().err
    assert "keys.txt line 1" in err and "too-short" not in err
