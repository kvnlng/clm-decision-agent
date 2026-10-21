"""Tests for chat.py, the test chat: an MCP client that shows each decision."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
from conftest import FakeRouter, decision
from mcp import Client

import server
from chat import format_decision, run_chat
from scenario import load_scenario
from tools import load_tools

DECISION = {
    "route": "Data_Access", "confidence": 0.91, "uncertain": False, "source": "clm",
    "probabilities": {"Data_Access": 0.91, "Information_Request": 0.05, "Out_Of_Scope": 0.04},
    "answers": {"identifiable": {"type": "noul", "noul": 0.88}, "urgency": {"type": "score", "score": 0.4},
                "team": {"type": "choice", "choice": "b", "confidence": 0.7, "probabilities": {}}},
    "rules": [{"when": "identifiable >= 0.5", "tell": "Add the label governance-review."}],
    "informed": [{"tool": "lookup_queue", "status": "done", "facts": ["Data Governance handles it.", "5 days."]}],
    "actions": [{"tool": "create_ticket", "status": "done", "result": {"number": 7}}],
}


def test_format_decision_layout():
    assert format_decision(DECISION) == [
        "  route: Data_Access 0.91 (next: Information_Request 0.05)",
        "  answers: identifiable 0.88   urgency 0.40   team b 0.70",
        "  facts: lookup_queue: Data Governance handles it. 5 days.",
        "  rules: identifiable >= 0.5: Add the label governance-review.",
        '  actions: create_ticket done: {"number": 7}',
    ]


def test_format_uncertain_and_skipped():
    uncertain = {**DECISION, "confidence": 0.35, "uncertain": True, "rules": [],
                 "probabilities": {"Data_Access": 0.35, "Information_Request": 0.3},
                 "informed": [{"tool": "lookup_queue", "status": "failed", "error": "OSError: no file"}],
                 "actions": [{"tool": "create_ticket", "status": "skipped", "reason": "route uncertain"}]}
    assert format_decision(uncertain) == [
        "  route: Data_Access 0.35 (next: Information_Request 0.30) uncertain",
        "  answers: identifiable 0.88   urgency 0.40   team b 0.70",
        "  facts: lookup_queue failed: OSError: no file",
        "  actions: create_ticket skipped: route uncertain",
    ]


def test_format_names_a_reused_correction():
    reviewed = {**DECISION, "source": "reviewed", "clm_route": "Out_Of_Scope", "answers": {}, "informed": [],
                "rules": [], "actions": []}
    assert format_decision(reviewed) == [
        "  route: Data_Access 0.91 (next: Information_Request 0.05) from your correction; CLM said Out_Of_Scope"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def chat(write_scenario):
    """-> run(lines, review) -> what the chat wrote, against a server whose CLM always answers Tech."""
    scenario = load_scenario(write_scenario())

    async def _run(lines, review=False):
        router = FakeRouter(decision("Tech", answers={"urgent": {"type": "noul", "noul": 0.2}}))
        written, typed = [], iter(lines)

        def read(_prompt):
            line = next(typed, None)
            if line is None:
                raise EOFError      # as input() does at the end of input
            return line
        async with Client(server.build_server(scenario, router, load_tools(scenario), review)) as client:
            await run_chat(client, read=read, write=written.append)
        return written
    _run.scenario = scenario
    return _run


@pytest.mark.anyio
async def test_chat_decides_typed_message(chat):
    written = await chat(["my app crashes", "/quit"])
    assert "  route: Tech 0.90 (next: Billing 0.10)" in written
    assert "  answers: urgent 0.20" in written


@pytest.mark.anyio
async def test_chat_ends_at_end_of_input(chat):
    assert await chat([]) == []


@pytest.mark.anyio
async def test_routes_command_shows_the_scenario(chat):
    written = await chat(["/routes", "/quit"])
    assert "  Billing: Charges and refunds" in written
    assert "  urgent (noul): Is this urgent?" in written


@pytest.mark.anyio
async def test_correct_saves_the_last_message(chat):
    written = await chat(["my app crashes", "/correct Billing", "/agreement", "/quit"], review=True)
    assert "Saved: the right route for that request is Billing." in written
    assert "CLM's own route agrees with 0 of your 1 corrections." in written
    assert chat.scenario.record_path.exists()


@pytest.mark.anyio
async def test_correct_needs_a_previous_message(chat):
    written = await chat(["/correct Billing", "/quit"], review=True)
    assert "Type a request first; /correct applies to the last one." in written


@pytest.mark.anyio
async def test_correct_unknown_route_says_so(chat):
    written = await chat(["my app crashes", "/correct Shipping", "/quit"], review=True)
    assert any(line.startswith("error:") and "unknown route 'Shipping'" in line for line in written)
    assert not chat.scenario.record_path.exists()


@pytest.mark.anyio
async def test_review_commands_absent_outside_review(chat):
    written = await chat(["my app crashes", "/correct Billing", "/agreement", "/quit"])
    assert written.count("That needs review mode: start the chat with --review.") == 2


@pytest.mark.anyio
async def test_unknown_command_lists_the_commands(chat):
    written = await chat(["/help", "/quit"])
    assert "Commands: /routes, /correct <route>, /agreement, /quit" in written
