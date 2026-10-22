"""Shared fixtures and model-free fakes for the tests."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import sys
import threading
from pathlib import Path

import pytest

from scenario import load_scenario

sys.path.append(str(Path(__file__).parent.parent / "examples" / "service_desk"))
sys.path.append(str(Path(__file__).parent.parent / "examples" / "crowdstrike"))

BASE_TOML = '''
[clm]
model_id = "test/clm"

[generator]
backend = "mlx"
model = "test/gen"
max_tokens = 50

[routing]
instructions = "Which workflow?"
confidence_threshold = 0.6

[tools]
module = "fake_tools"

[tools.inform.lookup]
questions = ["urgent"]

[routes.Billing]
description = "Charges and refunds"
inform = ["lookup"]

[routes.Tech]
description = "Bugs and outages"

[questions.urgent]
type = "noul"
instructions = "Is this urgent?"

[[actions]]
tool = "act"
routes = ["Billing"]
when = "urgent >= 0.5"
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


class FakeRouter:  # pylint: disable=too-many-instance-attributes
    """`decisions` is one decision used for every state, or a dict of state_text -> decision."""

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def __init__(self, decisions, tool_p=0.9, vectors=None, asks=None, ranked=None):
        self.decisions, self.tool_p, self.vectors = decisions, tool_p, vectors or {}
        # rank() answers with `ranked` (an option id and its confidence), or the first option at 0.9
        self.ranked, self.rank_calls = ranked, []
        self.states, self.tool_states, self.verdicts = [], [], {}
        # Answers for ask(), returned in order
        self.asks, self.ask_calls = list(asks or []), []

    def decide(self, state_text, scenario):
        self.states.append(state_text)
        found = dict(self.decisions if "route" in self.decisions else self.decisions[state_text])
        if scenario.live_corrections and state_text in self.verdicts:
            found.update(route=self.verdicts[state_text], source="reviewed", clm_route=found["route"])
        return found

    def prefetch(self, state_texts, questions):
        self.prefetched = getattr(self, "prefetched", []) + [(list(state_texts), questions)]

    def decide_batch(self, state_texts, scenario):
        self.prefetch(state_texts, "decide")
        return [self.decide(text, scenario) for text in state_texts]

    def route_batch(self, state_texts, scenario):
        self.prefetch(state_texts, "route")
        return [self.route(text, scenario) for text in state_texts]

    def route(self, state_text, scenario):
        found = self.decide(state_text, scenario)
        found.pop("answers")
        return found

    def rank(self, state_text, options, instructions):
        self.rank_calls.append((state_text, options, instructions))
        choice, confidence = self.ranked or (next(iter(options)), 0.9)
        rest = (1 - confidence) / max(len(options) - 1, 1)
        return {"choice": choice, "confidence": confidence,
                "probabilities": {key: confidence if key == choice else rest for key in options}}

    def add_verdict(self, state_text, route, scenario):
        self.verdicts[state_text] = route

    def check_tool(self, state_text, scenario):
        self.tool_states.append(state_text)
        return self.tool_p

    def embed_state(self, state_text, scenario):
        return self.vectors[state_text]

    def ask(self, state, questions):
        self.ask_calls.append((state, questions))
        return self.asks.pop(0)


class FakeGenerator:
    """Returns the given replies in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, system, user):
        self.calls.append({"system": system, "user": user})
        return self.replies.pop(0)


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
