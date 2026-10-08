"""Shared fixtures and model-free fakes for the tests."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import sys
import threading
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from scenario import load_scenario

sys.path.append(str(Path(__file__).parent.parent / "examples" / "service_desk"))

BASE_TOML = '''
[clm]
model_id = "test/clm"
history_turns = 2

[generator]
backend = "mlx"
model = "test/gen"
max_tokens = 50

[routing]
instructions = "Which workflow?"
confidence_threshold = 0.6

[tools]
module = "fake_tools"

[routes.Billing]
description = "Charges and refunds"
system_prompt = "You handle billing."
tools = ["lookup_invoice"]

[routes.Tech]
description = "Bugs and outages"
system_prompt = "You handle tech."

[questions.urgent]
type = "noul"
instructions = "Is this urgent?"
'''


@pytest.fixture
def write_scenario(tmp_path):
    def _write(text=BASE_TOML, name="scenario.toml"):
        path = tmp_path / name
        path.write_text(text)
        return path
    return _write


@pytest.fixture
def scenario(write_scenario):
    return load_scenario(write_scenario())


def decision(route, confidence=0.9, answers=None, probabilities=None):
    """A router decision for the BASE_TOML scenario (routes Billing and Tech)."""
    other = "Tech" if route == "Billing" else "Billing"
    return {"route": route, "confidence": confidence,
            "probabilities": probabilities or {route: 0.9, other: 0.1},
            "answers": answers or {}, "source": "clm"}


def tool_call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1"}])


class FakeRouter:
    """`decisions` is one decision used for every state, or a dict of state_text -> decision."""

    def __init__(self, decisions, tool_p=0.9, vectors=None):
        self.decisions, self.tool_p, self.vectors = decisions, tool_p, vectors or {}
        self.states, self.tool_states = [], []

    def decide(self, state_text, scenario):
        self.states.append(state_text)
        found = self.decisions if "route" in self.decisions else self.decisions[state_text]
        return dict(found)

    def check_tool(self, state_text, scenario):
        self.tool_states.append(state_text)
        return self.tool_p

    def embed_state(self, state_text, scenario):
        return self.vectors[state_text]


class FakeGenerator:
    """Returns the given replies in order; a str becomes a plain AIMessage."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, system, messages, tools=None):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        reply = self.replies.pop(0)
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply)


CONFIRM_TOML = (BASE_TOML
                .replace('tools = ["lookup_invoice"]', 'tools = ["lookup_invoice", "issue_refund"]')
                .replace('module = "fake_tools"', 'module = "fake_tools"\nconfirm = ["issue_refund"]'))


INTAKE_TOML = (BASE_TOML
               .replace('tools = ["lookup_invoice"]', 'tools = ["lookup_invoice", "submit_form", "broken_submit"]')
               .replace('module = "fake_tools"', 'module = "fake_tools"\nconfirm = ["submit_form", "broken_submit"]')
               + '\n[routes.Billing.fields]\ninvoice = "Which invoice"\nproblem = "What is wrong with it"\n')


@pytest.fixture
def tracker_server():
    from mock_tracker import make_server  # pylint: disable=import-outside-toplevel
    server = make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def tracker_url(tracker_server):
    return f"http://127.0.0.1:{tracker_server.server_address[1]}"
