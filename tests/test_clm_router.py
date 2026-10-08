"""Tests for clm_router.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
import pytest
from conftest import BASE_TOML

import clm_router
from clm_router import ROUTE_QID, ClmError, LocalContrastiveRouter, RemoteEngine, last_turn, make_router, resolve_heads
from scenario import load_scenario


class FakeEngine:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def answer(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": dict(self.answers)}


ROUTE_ANSWER = {"type": "choice", "choice": "Billing", "confidence": 0.8,
                "probabilities": {"Billing": 0.9, "Tech": 0.1}}
URGENT_ANSWER = {"type": "noul", "noul": 0.7}


def test_decide_sends_route_question_and_extras_in_one_call(scenario):
    engine = FakeEngine({ROUTE_QID: ROUTE_ANSWER, "urgent": URGENT_ANSWER})
    decision = LocalContrastiveRouter(engine=engine).decide("User: charged twice", scenario)

    assert len(engine.calls) == 1
    state, questions = engine.calls[0]
    assert state == "User: charged twice"
    assert questions[ROUTE_QID] == {
        "type": "choice",
        "instructions": "Which workflow?",
        "criteria": {"Billing": "Charges and refunds", "Tech": "Bugs and outages"},
    }
    assert questions["urgent"] == {"type": "noul", "instructions": "Is this urgent?"}
    assert decision == {
        "route": "Billing",
        "confidence": 0.8,
        "probabilities": {"Billing": 0.9, "Tech": 0.1},
        "answers": {"urgent": URGENT_ANSWER},
        "source": "clm",
    }


def test_check_tool_asks_one_noul_question(scenario):
    engine = FakeEngine({"tool": {"type": "noul", "noul": 0.82}})
    p = LocalContrastiveRouter(engine=engine).check_tool("User: check INV-1\n\nProposed tool call:\n- x({})", scenario)
    assert p == 0.82
    state, questions = engine.calls[0]
    assert state.endswith("- x({})")
    assert questions == {"tool": {"type": "noul",
                                  "instructions": "Is this tool call appropriate for what the user asked?"}}


class VectorRouter(LocalContrastiveRouter):
    """A router whose embeddings come from a table, so matching can be tested without a model."""

    def __init__(self, engine, vectors):
        super().__init__(engine=engine)
        self.vectors = vectors

    def embed_state(self, state_text, scenario):
        return self.vectors[state_text]


def correction(state_text, clm, human):
    return {"kind": "route", "state_text": state_text, "human": human,
            "clm": {"route": clm, "confidence": 0.8, "probabilities": {clm: 0.9}}}


VECTORS = {
    "User: bill me": [1.0, 0.0],
    "User: bill me please": [0.98, 0.199],   # cosine 0.98 with the first
    "User: app crashed": [0.0, 1.0],
}


def routed_to_billing():
    return FakeEngine({ROUTE_QID: ROUTE_ANSWER, "urgent": URGENT_ANSWER})


def test_close_correction_overrides_clm(scenario):
    router = VectorRouter(routed_to_billing(), VECTORS)
    assert router.load_verdicts([correction("User: bill me", "Billing", "Tech")], scenario) == 1

    decision = router.decide("User: bill me please", scenario)
    assert decision["route"] == "Tech"
    assert decision["source"] == "reviewed"
    assert decision["clm_route"] == "Billing"
    assert decision["similarity"] == pytest.approx(0.98)
    assert decision["confidence"] == 0.8


def test_distant_correction_is_ignored(scenario):
    router = VectorRouter(routed_to_billing(), VECTORS)
    router.load_verdicts([correction("User: bill me", "Billing", "Tech")], scenario)
    decision = router.decide("User: app crashed", scenario)
    assert (decision["route"], decision["source"]) == ("Billing", "clm")
    assert "similarity" not in decision


def test_later_correction_wins_a_tie(scenario):
    router = VectorRouter(routed_to_billing(), VECTORS)
    router.load_verdicts([correction("User: bill me", "Billing", "Tech"),
                             correction("User: bill me", "Tech", "Billing")], scenario)
    assert router.decide("User: bill me", scenario)["route"] == "Billing"


def test_live_corrections_can_be_switched_off(write_scenario):
    off = load_scenario(write_scenario(BASE_TOML + "\n[record]\nlive_corrections = false\n"))
    router = VectorRouter(routed_to_billing(), VECTORS)
    router.load_verdicts([correction("User: bill me", "Billing", "Tech")], off)
    assert router.decide("User: bill me", off)["source"] == "clm"


def test_embed_state_projects_context_plus_routing_question(scenario):
    class Encoder:
        def embed(self, texts):
            self.texts = texts
            return mx.array([[3.0, 4.0]]), 5

    class Heads:
        def project(self, embedding, which):
            self.which = which
            return mx.array([[0.6, 0.8]])

    engine = FakeEngine({})
    engine.encoder, engine.heads = Encoder(), Heads()
    vector = LocalContrastiveRouter(engine=engine).embed_state("User: hi ", scenario)

    assert engine.encoder.texts == ["User: hi\n\nWhich workflow?"]
    assert engine.heads.which == "state"
    assert vector == pytest.approx([0.6, 0.8])


def test_last_turn_is_the_final_user_message_with_its_continuation_lines():
    assert last_turn("User: a\nAssistant: b\nUser: c\nd") == "User: c\nd"
    assert last_turn("User: only") == "User: only"


def test_a_repeated_message_later_in_a_conversation_matches_a_first_turn_correction(scenario):
    later = "User: bill me\nAssistant: ok\nUser: bill me"
    router = VectorRouter(routed_to_billing(), {**VECTORS, later: [0.0, 1.0]})
    router.load_verdicts([correction("User: bill me", "Billing", "Tech")], scenario)
    decision = router.decide(later, scenario)
    assert (decision["route"], decision["source"]) == ("Tech", "reviewed")
    assert decision["similarity"] == pytest.approx(1.0)


def test_a_confirmed_route_is_reused_too(scenario):
    router = VectorRouter(routed_to_billing(), VECTORS)
    assert router.load_verdicts([correction("User: bill me", "Billing", "Billing")], scenario) == 1
    decision = router.decide("User: bill me", scenario)
    assert (decision["route"], decision["source"], decision["clm_route"]) == ("Billing", "reviewed", "Billing")


def test_heads_are_used_only_when_they_match_the_scenario(write_scenario):
    path = write_scenario(BASE_TOML.replace('model_id = "test/clm"', 'model_id = "test/clm"\nheads = "my.heads"'))
    scenario = load_scenario(path)
    folder, note = resolve_heads(scenario)
    assert folder is None and "train.py" in note

    scenario.clm_heads.mkdir()
    (scenario.clm_heads / "training.json").write_text(json.dumps({"routing_sha": "something else"}))
    folder, note = resolve_heads(scenario)
    assert folder is None and "changed since" in note

    facts = {"routing_sha": scenario.routing_sha, "clm_model_id": "test/clm"}
    (scenario.clm_heads / "training.json").write_text(json.dumps(facts))
    folder, note = resolve_heads(scenario)
    assert folder is None and "weights file is missing" in note

    (scenario.clm_heads / "CLM_v0.1-8B.safetensors").write_bytes(b"x")
    assert resolve_heads(scenario) == (scenario.clm_heads, "trained heads from my.heads")

    (scenario.clm_heads / "training.json").write_text(json.dumps({**facts, "clm_model_id": "other/clm"}))
    folder, note = resolve_heads(scenario)
    assert folder is None and "different CLM build" in note

    (scenario.clm_heads / "training.json").write_text("[]")
    assert resolve_heads(scenario)[0] is None


def test_no_heads_setting_means_the_released_heads(scenario):
    assert resolve_heads(scenario) == (None, "released heads")


@pytest.fixture
def clm_server():
    """A stand-in for the CLM authors' server: POST /v1/systemone answers every question with its first option."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # pylint: disable=invalid-name
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.seen.append((self.path, self.headers.get("Authorization"), body))
            if self.server.key and self.headers.get("Authorization") != f"Bearer {self.server.key}":
                status, out = 401, {"detail": "invalid API key"}
            else:
                answers = {}
                for name, question in body["questions"].items():
                    if question["type"] == "choice":
                        first = next(iter(question["criteria"]))
                        answers[name] = {"type": "choice", "choice": first, "confidence": 0.9,
                                         "probabilities": {key: 0.9 if key == first else 0.1
                                                           for key in question["criteria"]}}
                    else:
                        answers[name] = {"type": "noul", "noul": 0.7}
                status, out = 200, {"model": body.get("model"), "answers": answers, "usage": {}}
            payload = json.dumps(out).encode()
            self.send_response(status)
            for header, value in (("Content-Type", "application/json"), ("Content-Length", str(len(payload)))):
                self.send_header(header, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # pylint: disable=redefined-builtin
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.seen, server.key = [], None
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def server_scenario(write_scenario, server, extra=""):
    url = f"http://127.0.0.1:{server.server_address[1]}"
    text = BASE_TOML.replace('model_id = "test/clm"', f'backend = "server"\nurl = "{url}"\n{extra}')
    return load_scenario(write_scenario(text))


def test_server_backend_decides_and_checks_tools_over_http(write_scenario, clm_server):
    scenario = server_scenario(write_scenario, clm_server)
    router, note = make_router(scenario)
    decision = router.decide("User: charged twice", scenario)

    assert "127.0.0.1" in note
    assert (decision["route"], decision["confidence"], decision["source"]) == ("Billing", 0.9, "clm")
    assert decision["answers"] == {"urgent": {"type": "noul", "noul": 0.7}}
    path, authorization, body = clm_server.seen[0]
    assert (path, authorization) == ("/v1/systemone", None)
    assert body["state"] == "User: charged twice" and body["model"] == "clm-latest"
    assert body["questions"][ROUTE_QID]["criteria"] == {"Billing": "Charges and refunds", "Tech": "Bugs and outages"}
    assert router.check_tool("User: x", scenario) == 0.7


def test_server_backend_sends_the_key_as_a_bearer_token(write_scenario, clm_server, monkeypatch):
    clm_server.key = "s3cret"
    monkeypatch.setenv("DEMO_CLM_KEY", "s3cret")
    scenario = server_scenario(write_scenario, clm_server, 'api_key_env = "DEMO_CLM_KEY"')
    assert make_router(scenario)[0].decide("User: hi", scenario)["route"] == "Billing"
    assert clm_server.seen[0][1] == "Bearer s3cret"


def test_server_backend_reports_a_rejected_key_and_an_unreachable_server(write_scenario, clm_server):
    clm_server.key = "s3cret"
    scenario = server_scenario(write_scenario, clm_server)
    with pytest.raises(ClmError, match="rejected the API key"):
        make_router(scenario)[0].decide("User: hi", scenario)
    with pytest.raises(ClmError, match="http://127.0.0.1:9"):
        RemoteEngine("http://127.0.0.1:9", "clm-latest").answer("User: hi", {"q": {"type": "noul"}})


def test_server_backend_reuses_a_reviewed_route_when_the_text_is_the_same(write_scenario, clm_server):
    scenario = server_scenario(write_scenario, clm_server)
    router, _ = make_router(scenario)
    records = [{"kind": "route", "state_text": "User: my app crashes", "human": "Tech",
                "clm": {"route": "Billing", "confidence": 0.9, "probabilities": {}}}]
    assert router.load_verdicts(records, scenario) == 1

    same = router.decide("User: my app crashes", scenario)
    assert (same["route"], same["source"]) == ("Tech", "reviewed")
    assert (same["clm_route"], same["similarity"]) == ("Billing", 1.0)
    later = router.decide("User: hello\nAssistant: hi\nUser: my app crashes", scenario)
    assert later["route"] == "Tech"
    assert router.decide("User: my app crashed", scenario)["source"] == "clm"


def test_trained_heads_are_not_used_with_the_server_backend(write_scenario, clm_server):
    scenario = server_scenario(write_scenario, clm_server, 'heads = "my.heads"')
    assert "trained heads apply to the mlx backend only" in make_router(scenario)[1]


@pytest.mark.parametrize("body", [
    b"<html>proxy error</html>", b'{"detail": "oops"}', b"[1, 2]",
    json.dumps({"answers": {"other": {"type": "noul", "noul": 0.5}}}).encode(),
    json.dumps({"answers": {"q": {"type": "choice", "choice": "Nope", "confidence": 0.9,
                                  "probabilities": {"Nope": 1.0}}}}).encode(),
    json.dumps({"answers": {"q": {"type": "choice", "choice": "a"}}}).encode(),
])
def test_a_server_reply_that_cannot_be_used_is_a_clm_error(monkeypatch, body):
    monkeypatch.setattr(clm_router.httpx, "post", lambda *args, **kwargs: clm_router.httpx.Response(200, content=body))
    question = {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}}}
    with pytest.raises(ClmError, match="could not be used"):
        RemoteEngine("http://x", "clm-latest").answer("User: hi", question)


def test_server_errors_do_not_repeat_what_the_server_sent(monkeypatch):
    monkeypatch.setattr(clm_router.httpx, "post",
                        lambda *args, **kwargs: clm_router.httpx.Response(500, text="Bearer s3cret was wrong"))
    with pytest.raises(ClmError) as failure:
        RemoteEngine("http://x", "clm-latest", "s3cret").answer("User: hi", {"q": {"type": "noul"}})
    assert "HTTP 500" in str(failure.value) and "s3cret" not in str(failure.value)
