"""Shared fixtures and model-free fakes for the tests."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import contextlib
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from clm_decision.scenario import load_scenario

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

    def prefetch_asks(self, asks, ranks=()):
        self.prefetched = getattr(self, "prefetched", []) + [([state for state, _ in asks], list(ranks))]

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


# A small environment: two suppliers of one product, one calculation and one limit that rejects moving everything
ENV_TOML = '''
[environment]
name = "Toy sourcing"
description = "Where to buy one product's parts"
functions = "toy_functions"

[graph]
nodes = "graph/nodes.csv"
edges = "graph/edges.csv"

[queries.suppliers]
description = "The product's suppliers, with region and unit cost"
params = { product = "string" }
function = "suppliers"

[decisions.share_limit]
description = "Whether a share for Region B is allowed"
dmn = "rules/share_limit.dmn"

[calculations.cost]
description = "Unit cost and margin for a share"
params = { share = "number", cost_a = "number", cost_b = "number" }
function = "cost"

[scenarios.sourcing]
description = "How much to buy from Region B"
params = { product = "string" }
levers = { share = [0, 0.5, 1.0] }
branch = "branch"
objective = "margin"
limits = ["share_limit"]
'''

ENV_NODES = "id,kind,name,region,cost\nP1,product,Widget,,\nS1,supplier,Alpha,A,10\nS2,supplier,Beta,B,6\n"
ENV_EDGES = "source,relation,target\nS1,supplies,P1\nS2,supplies,P1\n"

ENV_FUNCTIONS = '''
CALLS = []


def suppliers(graph, product):
    return [{"id": node["id"], "region": node["region"], "cost": node["cost"]}
            for node in graph.neighbours(product, "supplies", "in")]


def cost(share, cost_a, cost_b):
    unit = cost_a * (1 - share) + cost_b * share
    return {"unit_cost": unit, "margin": 20 - unit}


def branch(steps, params, levers):
    CALLS.append(dict(levers))
    costs = {row["region"]: row["cost"] for row in steps.query("suppliers", product=params["product"])}
    figures = steps.calculate("cost", share=levers["share"], cost_a=costs["A"], cost_b=costs["B"])
    steps.decide("share_limit", share=levers["share"])
    return figures


def branch_swallowing(steps, params, levers):
    figures = branch(steps, params, levers)
    try:
        steps.decide("share_limit", share=0.05)
    except ValueError:
        pass
    return figures


def branch_without_limit(steps, params, levers):
    costs = {row["region"]: row["cost"] for row in steps.query("suppliers", product=params["product"])}
    return steps.calculate("cost", share=levers["share"], cost_a=costs["A"], cost_b=costs["B"])
'''


def limit_dmn(allowed: str = "&lt;= 0.6") -> str:
    """The share_limit table: shares above 0.9 are refused, shares matching `allowed` are allowed."""
    rule = ('<rule id="{id}"><inputEntry><text>{test}</text></inputEntry><outputEntry><text>{allowed}</text>'
            '</outputEntry><outputEntry><text>"{reason}"</text></outputEntry></rule>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" id="d" name="d" namespace="t">'
            '<decision id="share_limit" name="share_limit"><decisionTable id="t">'
            '<input id="i"><inputExpression typeRef="number"><text>share</text></inputExpression></input>'
            '<output id="o1" name="allowed" typeRef="boolean"/><output id="o2" name="reason" typeRef="string"/>'
            + rule.format(id="too_much", test="&gt; 0.9", allowed="false", reason="Too much from Region B")
            + rule.format(id="fine", test=allowed, allowed="true", reason="Within the limit")
            + "</decisionTable></decision></definitions>")


@pytest.fixture
def write_environment(tmp_path):
    def _write(toml=ENV_TOML, functions=ENV_FUNCTIONS, dmn=None, folder="toy"):
        root = tmp_path / folder
        (root / "graph").mkdir(parents=True)
        (root / "rules").mkdir()
        (root / "environment.toml").write_text(toml, encoding="utf-8")
        (root / "toy_functions.py").write_text(functions, encoding="utf-8")
        (root / "graph" / "nodes.csv").write_text(ENV_NODES, encoding="utf-8")
        (root / "graph" / "edges.csv").write_text(ENV_EDGES, encoding="utf-8")
        (root / "rules" / "share_limit.dmn").write_text(dmn or limit_dmn(), encoding="utf-8")
        return root / "environment.toml"
    return _write


JUDGE_TOML = '''
[clm]
backend = "mlx"

[judges.risk]
description = "Whether a piece of news is a delivery risk"
instructions = "Does this news mean the supplier may fail to deliver?"
'''


class FakeClmEngine:
    """CLM's engine for judges: each text's probability from `probabilities`, by a word it contains, else 0.1."""

    def __init__(self, probabilities=None):
        self.probabilities = probabilities or {}
        self.asked = []

    def answer(self, state, questions):
        self.asked.append((state, questions))
        probability = next((p for word, p in self.probabilities.items() if word in state), 0.1)
        return {"answers": {name: {"type": "noul", "noul": probability} for name in questions}}


@pytest.fixture
def fake_clm(monkeypatch):
    """Replaces CLM's loading; -> the engine judges get, with `loads` counting how often it was loaded."""
    from clm_decision import clm_router  # pylint: disable=import-outside-toplevel
    engine = FakeClmEngine({"strike": 0.9})
    engine.loads = 0

    def load(backend, model_id, heads_dir=None, device=None):
        engine.loads += 1
        return engine
    monkeypatch.setattr(clm_router, "load_engine", load)
    return engine


def free_port() -> int:
    """A port nothing is listening on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextlib.contextmanager
def serving(app, port: int):
    """Serve an ASGI app on 127.0.0.1:port for the block."""
    import uvicorn  # pylint: disable=import-outside-toplevel
    runner = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    while not runner.started:
        time.sleep(0.02)
    try:
        yield
    finally:
        runner.should_exit = True
        thread.join()
