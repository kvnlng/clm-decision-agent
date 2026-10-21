"""Tests for clm_router.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import json

import numpy as np
import pytest
from conftest import BASE_TOML

import clm_router
from clm_router import ROUTE_QID, LocalContrastiveRouter, make_router, resolve_heads
from scenario import load_scenario


class FakeEngine:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def answer(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": dict(self.answers)}

    def prefetch(self, states, questions):
        self.prefetched = (list(states), questions)


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


def test_an_added_verdict_is_reused_at_once(scenario):
    router = VectorRouter(routed_to_billing(), VECTORS)
    router.add_verdict("User: bill me", "Tech", scenario)
    decision = router.decide("User: bill me please", scenario)
    assert (decision["route"], decision["source"]) == ("Tech", "reviewed")


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
    class Engine(FakeEngine):
        def embed(self, texts):
            self.texts = texts
            return np.array([[3.0, 4.0]], dtype=np.float32)

        def project(self, vectors, which):
            self.which = which
            return np.array([[0.6, 0.8]], dtype=np.float32)

    engine = Engine({})
    vector = LocalContrastiveRouter(engine=engine).embed_state("User: hi ", scenario)

    assert engine.texts == ["User: hi\n\nWhich workflow?"]
    assert engine.which == "state"
    assert vector == pytest.approx([0.6, 0.8])


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


def heads_scenario(write_scenario, backend="mlx"):
    clm = f'backend = "{backend}"\nmodel_id = "test/clm"\nheads = "my.heads"'
    scenario = load_scenario(write_scenario(BASE_TOML.replace('model_id = "test/clm"', clm)))
    scenario.clm_heads.mkdir()
    (scenario.clm_heads / "CLM_v0.1-8B.safetensors").write_bytes(b"x")
    return scenario


def test_heads_trained_on_the_other_backend_are_not_used(write_scenario):
    scenario = heads_scenario(write_scenario, "torch")
    facts = {"routing_sha": scenario.routing_sha, "clm_model_id": "test/clm"}     # no backend: made with mlx
    (scenario.clm_heads / "training.json").write_text(json.dumps(facts))
    assert resolve_heads(scenario) == (None, "released heads (my.heads was trained with the mlx backend; "
                                             "run train.py again)")
    (scenario.clm_heads / "training.json").write_text(json.dumps({**facts, "clm_backend": "torch"}))
    assert resolve_heads(scenario) == (scenario.clm_heads, "trained heads from my.heads")


def test_make_router_passes_backend_and_device_to_load_engine(write_scenario, monkeypatch):
    seen = []
    monkeypatch.setattr(clm_router, "load_engine", lambda *args: seen.append(args) or FakeEngine({}))
    text = BASE_TOML.replace('model_id = "test/clm"', 'backend = "torch"\nmodel_id = "test/clm"\ndevice = "cpu"')
    router, note = make_router(load_scenario(write_scenario(text)))
    assert seen == [("torch", "test/clm", None, "cpu")]
    assert isinstance(router, LocalContrastiveRouter)
    assert note == "its released heads"


def test_route_asks_only_the_routing_question(scenario):
    engine = FakeEngine({ROUTE_QID: ROUTE_ANSWER})
    routed = LocalContrastiveRouter(engine=engine).route("User: charged twice\n\nKnown:\n- Fare A", scenario)
    assert routed == {"route": "Billing", "confidence": 0.8, "probabilities": {"Billing": 0.9, "Tech": 0.1},
                      "source": "clm"}
    state, questions = engine.calls[0]
    assert state.endswith("- Fare A") and list(questions) == [ROUTE_QID]


def test_rank_asks_one_choice_question_over_the_options():
    engine = FakeEngine({"option": {"type": "choice", "choice": "b", "confidence": 0.7,
                                    "probabilities": {"a": 0.3, "b": 0.7}}})
    ranked = LocalContrastiveRouter(engine=engine).rank("Late again", {"a": "Flight A", "b": "Flight B"}, "Which?")
    assert ranked == {"choice": "b", "confidence": 0.7, "probabilities": {"a": 0.3, "b": 0.7}}
    assert engine.calls == [("Late again", {"option": {"type": "choice", "instructions": "Which?",
                                                       "criteria": {"a": "Flight A", "b": "Flight B"}}})]


def test_a_single_option_is_chosen_with_full_confidence():
    from test_clm_torch import CONFIG, FakeEncoder, tiny_weights  # pylint: disable=import-outside-toplevel
    from clm_torch import TorchEngine, TorchHeads  # pylint: disable=import-outside-toplevel
    router = LocalContrastiveRouter(engine=TorchEngine(FakeEncoder(), TorchHeads(tiny_weights(), CONFIG, "cpu")))
    assert router.rank("Late again", {"a": "Flight A"}, "Which?")["confidence"] == pytest.approx(1.0)


def test_decide_batch_matches_decide_one_by_one(scenario):
    engine = FakeEngine({ROUTE_QID: ROUTE_ANSWER, "urgent": URGENT_ANSWER})
    router = LocalContrastiveRouter(engine=engine)
    batch = router.decide_batch(["one", "two"], scenario)
    assert engine.prefetched[0] == ["one", "two"] and set(engine.prefetched[1]) == {ROUTE_QID, "urgent"}
    assert batch == [router.decide("one", scenario), router.decide("two", scenario)]


class FakeMlxBuild:
    """The MLX build's Engine as far as prefetch uses it."""

    def __init__(self):
        self.vectors, self.cache, self.tokens_spent = [], {}, 0

    def _vectors(self, texts, which):
        self.vectors.append((which, list(texts)))


def test_mlx_prefetch_uses_the_builds_vectors():
    engine = clm_router.MlxEngine.__new__(clm_router.MlxEngine)
    engine.engine = FakeMlxBuild()
    engine.prefetch(["a", "b"], {"q": {"type": "noul", "instructions": "Urgent?"}})
    assert ("state", ["a\n\nUrgent?", "b\n\nUrgent?"]) in engine.engine.vectors
