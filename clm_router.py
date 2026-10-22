"""CLM decisions for a scenario. CLM scores a state against options; it does not generate text."""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download

import clm_schema
import tracing
from clm_common import HEADS_FILE, ClmError
from record import route_verdicts
from scenario import Scenario

ROUTE_QID = "__route__"


class MlxEngine:
    """The MLX build's engine behind the interface the router and train.py use, with numpy in and out."""

    def __init__(self, engine, heads_dir: str):
        self.engine = engine
        self.head_config = json.loads(Path(heads_dir, "config.json").read_text(encoding="utf-8"))

    def answer(self, state, questions: dict[str, dict]) -> dict:
        """CLM's answers to typed questions about a state."""
        if tracing.active():
            # The build encodes and projects inside answer(), so its cache is checked here for the trace
            pairs = clm_schema.build_pairs(state, questions).values()
            texts = [("state", pair[0]) for pair in pairs] + [("action", text) for pair in pairs for text in pair[2]]
            missing = {key for key in texts if key not in self.engine.cache}
            tracing.add("clm_calls")
            tracing.add("cache_hits", len(texts) - len(missing))
            tracing.add("cache_misses", len(missing))
            tracing.add("texts_encoded", len(missing))
        before = self.engine.tokens_spent
        with tracing.timed("encode_ms"):
            reply = self.engine.answer(state, questions)
        tracing.add("tokens", self.engine.tokens_spent - before)
        return reply

    def prefetch(self, states: list, questions: dict[str, dict]) -> None:
        """Encode, in one batched pass, every state text that answering `questions` about `states` will need, so
        those answers are found in the build's cache."""
        self.prefetch_asks([(state, questions) for state in states])

    def prefetch_asks(self, asks: list[tuple[Any, dict[str, dict]]]) -> None:
        """prefetch() for [(state, questions)], each state with its own questions, in one batched pass."""
        if not asks:
            return
        pairs = [pair for state, questions in asks for pair in clm_schema.build_pairs(state, questions).values()]
        texts = {"state": [pair[0] for pair in pairs], "action": [text for pair in pairs for text in pair[2]]}
        if tracing.active():
            tracing.add("texts_encoded", len({(which, text) for which, found in texts.items() for text in found
                                              if (which, text) not in self.engine.cache}))
        before = self.engine.tokens_spent
        # _vectors is the build's own; the model revision pins the build, so it does not change under us
        with tracing.timed("encode_ms"):
            for which, found in texts.items():
                self.engine._vectors(found, which)  # pylint: disable=protected-access
        tracing.add("tokens", self.engine.tokens_spent - before)

    def embed(self, texts: list[str]) -> np.ndarray:
        """[n, hidden] encoder vectors, unit length."""
        with tracing.timed("encode_ms"):
            vectors, tokens = self.engine.encoder.embed(texts)
            vectors = np.array(vectors)
        tracing.add("texts_encoded", len(texts))
        tracing.add("tokens", tokens)
        return vectors

    def project(self, vectors: np.ndarray, which: str) -> np.ndarray:
        """Encoder vectors through the "state" or "action" head."""
        import mlx.core as mx  # pylint: disable=import-outside-toplevel,import-error
        with tracing.timed("head_ms"):
            return np.array(self.engine.heads.project(mx.array(np.asarray(vectors, dtype=np.float32)), which))

    def head_weights(self) -> dict[str, np.ndarray]:
        """The current heads' weights, in the layout of a heads file."""
        from mlx.utils import tree_flatten  # pylint: disable=import-outside-toplevel,import-error
        heads = self.engine.heads
        weights = {"logit_scale": np.array([math.log(heads.scale)], dtype=np.float32)}
        for prefix, head in (("state_head", heads.state), ("action_head", heads.action)):
            weights.update({f"{prefix}.{name}": np.array(value) for name, value in tree_flatten(head.parameters())})
        return weights

    def set_head_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Use these head weights from now on."""
        import mlx.core as mx  # pylint: disable=import-outside-toplevel,import-error
        heads = self.engine.heads
        for prefix, head in (("state_head", heads.state), ("action_head", heads.action)):
            head.load_weights([(name[len(prefix) + 1:], mx.array(np.asarray(value, dtype=np.float32)))
                               for name, value in weights.items() if name.startswith(prefix + ".")], strict=True)
            mx.eval(head.parameters())
        heads.scale = min(math.exp(float(np.asarray(weights["logit_scale"]).reshape(-1)[0])), 100.0)
        self.engine.cache.clear()


def load_mlx_engine(model_id: str, heads_dir=None) -> MlxEngine:
    """Download (or reuse the cached) MLX build of CLM and return its engine.

    `heads_dir` swaps the build's own heads for ones trained with train.py; the encoder is the same."""
    logging.getLogger("transformers").setLevel(logging.ERROR)
    model_dir = snapshot_download(repo_id=model_id)
    # The clm_mlx package ships inside the model repo rather than on PyPI
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from clm_mlx.engine import Engine  # pylint: disable=import-outside-toplevel,import-error
    # truncation="tail" keeps the most recent tokens when a state exceeds 2048 tokens
    heads = str(heads_dir) if heads_dir else os.path.join(model_dir, "heads")
    return MlxEngine(Engine(os.path.join(model_dir, "encoder"), heads, truncation="tail"), heads)


def load_engine(backend: str, model_id: str, heads_dir=None, device: str | None = None):
    """CLM's engine for the backend: "mlx" (an Apple Silicon Mac) or "torch" (a GPU, or any machine slowly)."""
    if backend == "torch":
        from clm_torch import load_torch_engine  # pylint: disable=import-outside-toplevel
        return load_torch_engine(model_id, heads_dir, device)
    try:
        import mlx.core  # pylint: disable=import-outside-toplevel,import-error,unused-import
    except ImportError as error:
        raise ClmError('The mlx backend needs an Apple Silicon Mac. On other machines set backend = "torch" '
                       "under [clm]; see docs/models.md.") from error
    return load_mlx_engine(model_id, heads_dir)


def _stale(folder: Path, facts: dict, scenario: Scenario) -> str | None:
    """Why the trained heads in `folder` do not fit the scenario, or None if they do."""
    if facts.get("routing_sha") != scenario.routing_sha:
        return f"the routes or routing question changed since {folder.name} was trained; run train.py again"
    if facts.get("clm_backend", "mlx") != scenario.clm_backend:
        return f"{folder.name} was trained with the {facts.get('clm_backend', 'mlx')} backend; run train.py again"
    if facts.get("clm_model_id") != scenario.clm_model_id:
        return (f"{folder.name} was trained for a different CLM build, {facts.get('clm_model_id')}; "
                f"run train.py again")
    # Heads trained before questions were trained have no questions_sha: they learned the route alone
    for name, sha in facts.get("questions_sha", {}).items():
        if name.startswith("options of ") and name[len("options of "):] not in scenario.routes:
            continue
        if sha != scenario.question_sha(name):
            what = f"the {name}" if name.startswith("options of ") else f"the question '{name}'"
            return f"{what} changed since {folder.name} was trained; run train.py again"
    if not (folder / HEADS_FILE).is_file():
        return f"the weights file is missing from {folder.name}; run train.py again"
    return None


def resolve_heads(scenario: Scenario):
    """-> (folder of trained heads to use, or None for the released ones; a line saying which and why)."""
    folder = scenario.clm_heads
    if folder is None:
        return None, "released heads"
    try:
        facts = json.loads((folder / "training.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        facts = None
    if not isinstance(facts, dict):
        return None, (f"released heads ({folder.name} has no trained heads yet; "
                      f"create them with synthesize.py and train.py)")
    reason = _stale(folder, facts, scenario)
    return (None, f"released heads ({reason})") if reason else (folder, f"trained heads from {folder.name}")


def make_router(scenario: Scenario):
    """-> (the router for the scenario's CLM backend, a line saying which heads it is using)."""
    heads_dir, note = resolve_heads(scenario)
    engine = load_engine(scenario.clm_backend, scenario.clm_model_id, heads_dir, scenario.clm_device)
    return LocalContrastiveRouter(engine=engine), f"its {note}"


def _option_question(options: dict[str, str], instructions: str) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def _routing_question(scenario: Scenario) -> dict:
    return {"type": "choice", "instructions": scenario.routing_instructions,
            "criteria": {name: route.description for name, route in scenario.routes.items()}}


class LocalContrastiveRouter:
    """CLM for one scenario: routing decisions, tool-call checks and reuse of reviewed routes."""

    def __init__(self, engine):
        self.engine = engine
        self.verdicts: list = []

    def decide(self, state_text: str, scenario: Scenario) -> dict:
        """One CLM call: the routing question plus the scenario's extra questions."""
        questions = {ROUTE_QID: _routing_question(scenario), **scenario.questions}
        answers = self.engine.answer(state_text, questions)["answers"]
        return {**self._routed(state_text, answers.pop(ROUTE_QID), scenario), "answers": answers}

    def prefetch(self, state_texts: list[str], questions: dict[str, dict]) -> None:
        """Encode many states in one batched pass, ahead of the decisions that will ask `questions` of them."""
        self.engine.prefetch(state_texts, questions)

    def decide_batch(self, state_texts: list[str], scenario: Scenario) -> list[dict]:
        """decide() for each state, with all of them encoded in one batched pass first."""
        self.prefetch(state_texts, {ROUTE_QID: _routing_question(scenario), **scenario.questions})
        return [self.decide(text, scenario) for text in state_texts]

    def route_batch(self, state_texts: list[str], scenario: Scenario) -> list[dict]:
        """route() for each state, with all of them encoded in one batched pass first."""
        self.prefetch(state_texts, {ROUTE_QID: _routing_question(scenario)})
        return [self.route(text, scenario) for text in state_texts]

    def route(self, state_text: str, scenario: Scenario) -> dict:
        """The routing question alone: {route, confidence, probabilities, source}, and the reviewed route if one
        matches."""
        answers = self.engine.answer(state_text, {ROUTE_QID: _routing_question(scenario)})["answers"]
        return self._routed(state_text, answers[ROUTE_QID], scenario)

    def prefetch_asks(self, asks: list[tuple[str, dict]], ranks: list[tuple[str, dict[str, str], str]] = ()) -> None:
        """Encode, in one batched pass, what answering `asks` ([(state text, questions)]) and ranking `ranks`
        ([(state text, options, instructions)], as rank() takes them) will need."""
        self.engine.prefetch_asks([*asks, *((state, {"option": _option_question(options, instructions)})
                                            for state, options, instructions in ranks)])

    def rank(self, state_text: str, options: dict[str, str], instructions: str) -> dict:
        """Which of `options` ({id: description}) fits the state: {choice, confidence, probabilities}."""
        question = _option_question(options, instructions)
        answer = self.engine.answer(state_text, {"option": question})["answers"]["option"]
        return {key: answer[key] for key in ("choice", "confidence", "probabilities")}

    def _routed(self, state_text: str, route: dict, scenario: Scenario) -> dict:
        decision = {
            "route": route["choice"],
            "confidence": route["confidence"],
            "probabilities": route["probabilities"],
            "source": "clm",
        }
        if self.verdicts and scenario.live_corrections:
            match = self._reviewed(state_text, scenario)
            if match:
                human_route, similarity = match
                decision.update(route=human_route, source="reviewed", similarity=similarity,
                                clm_route=route["choice"])
        return decision

    def _reviewed(self, state_text: str, scenario: Scenario) -> tuple[str, float] | None:
        """The route a person chose for this state before, if one matches: (route, similarity)."""
        return self.match_verdict(self.embed_state(state_text, scenario), scenario.match_threshold)

    def embed_state(self, state_text: str, scenario: Scenario) -> list[float]:
        """CLM's state-head vector for this state under the routing question (unit length)."""
        # Same layout as clm_mlx.schema.state_text: context, blank line, question
        text = f"{state_text.strip()}\n\n{scenario.routing_instructions.strip()}"
        return self.engine.project(self.engine.embed([text]), "state")[0].tolist()

    def load_verdicts(self, records: list[dict], scenario: Scenario) -> int:
        """Embed the reviewed routes in `records` for matching in decide(); returns how many."""
        self.verdicts = [(self.embed_state(r["state_text"], scenario), r["human"])
                         for r in route_verdicts(records, scenario)]
        return len(self.verdicts)

    def add_verdict(self, state_text: str, route: str, scenario: Scenario) -> None:
        """Reuse the owner's `route` for this state (and near-identical ones) from now on."""
        self.verdicts.append((self.embed_state(state_text, scenario), route))

    def match_verdict(self, vector: list[float], threshold: float) -> tuple[str, float] | None:
        """The closest stored verdict at or above the threshold; on a tie the later one wins."""
        best_route, best_similarity = None, threshold
        with tracing.timed("match_ms"):
            for stored, human_route in self.verdicts:
                similarity = sum(a * b for a, b in zip(vector, stored))
                if similarity >= best_similarity:
                    best_route, best_similarity = human_route, similarity
        return None if best_route is None else (best_route, best_similarity)

    def check_tool(self, state_text: str, scenario: Scenario) -> float:
        """Probability that a proposed tool call is appropriate; the call is described in state_text."""
        question = {"type": "noul", "instructions": scenario.tool_check_instructions}
        return self.engine.answer(state_text, {"tool": question})["answers"]["tool"]["noul"]

    def ask(self, state: str, questions: dict[str, dict]) -> dict:
        """Typed questions (noul / choice / score) answered against one state in a single pass."""
        return self.engine.answer(state, questions)["answers"]
