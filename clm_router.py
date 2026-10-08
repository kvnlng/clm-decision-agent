"""CLM decisions for a scenario. CLM scores a state against options; it does not generate text."""
from __future__ import annotations

import json
import logging
import os
import sys

import httpx
from huggingface_hub import snapshot_download

from record import route_verdicts
from scenario import DEFAULT_CLM, Scenario

ROUTE_QID = "__route__"
HEADS_FILE = "CLM_v0.1-8B.safetensors"     # the name the released heads use, so trained ones load the same way


def load_engine(model_id: str, heads_dir=None):
    """Download (or reuse the cached) CLM build and return its clm_mlx Engine.

    `heads_dir` swaps the build's own heads for ones trained with train.py; the encoder is the same."""
    logging.getLogger("transformers").setLevel(logging.ERROR)
    model_dir = snapshot_download(repo_id=model_id)
    # The clm_mlx package ships inside the model repo rather than on PyPI
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from clm_mlx.engine import Engine  # pylint: disable=import-outside-toplevel,import-error
    # truncation="tail" keeps the most recent tokens when a state exceeds 2048 tokens
    heads = str(heads_dir) if heads_dir else os.path.join(model_dir, "heads")
    return Engine(os.path.join(model_dir, "encoder"), heads, truncation="tail")


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
    if facts.get("routing_sha") != scenario.routing_sha:
        return None, (f"released heads (the routes or routing question changed since {folder.name} was trained; "
                      f"run train.py again)")
    if facts.get("clm_model_id") != scenario.clm_model_id:
        return None, (f"released heads ({folder.name} was trained for a different CLM build, "
                      f"{facts.get('clm_model_id')}; run train.py again)")
    if not (folder / HEADS_FILE).is_file():
        return None, f"released heads (the weights file is missing from {folder.name}; run train.py again)"
    return folder, f"trained heads from {folder.name}"


class ClmError(RuntimeError):
    """CLM could not be asked; the message says what to do about it."""


class RemoteEngine:
    """Asks the CLM authors' server (clm-serve) instead of a model in this process.

    It offers the same answer(state, questions) as the local engine, through POST /v1/systemone.
    The server returns answers, not vectors, so nothing here can embed a state."""

    def __init__(self, url: str, model: str, api_key: str | None = None):
        self.url, self.model, self.api_key = url.rstrip("/"), model, api_key

    def answer(self, state: str, questions: dict[str, dict]) -> dict:
        """The server's answers to the questions about this state, in the local engine's shape."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"state": state, "model": self.model, "questions": questions}
        try:
            response = httpx.post(f"{self.url}/v1/systemone", json=body, headers=headers, timeout=60)
        except httpx.HTTPError as error:
            raise ClmError(f"Cannot reach the CLM server at {self.url} ({type(error).__name__}).") from error
        if response.status_code == 401:
            raise ClmError(f"The CLM server at {self.url} rejected the API key. Check the variable named by "
                           f"api_key_env under [clm].")
        if response.status_code != 200:
            # The body is not repeated: a server may echo the request, key included
            raise ClmError(f"The CLM server at {self.url} answered HTTP {response.status_code}.")
        try:
            reply = response.json()
            self._check(reply["answers"], questions)
        except (ValueError, KeyError, TypeError) as error:
            raise ClmError(f"The CLM server at {self.url} sent a reply that could not be used "
                           f"({type(error).__name__}: {error}).") from error
        return reply

    @staticmethod
    def _check(answers: dict, questions: dict[str, dict]) -> None:
        """Raise unless every question has an answer of its own type that the router can read."""
        for name, question in questions.items():
            answer = answers[name]
            if question["type"] == "choice":
                if answer["choice"] not in question["criteria"]:
                    raise ValueError(f"'{answer['choice']}' is not one of the options for {name}")
                float(answer["confidence"])
                dict(answer["probabilities"])
            else:
                float(answer["noul" if question["type"] == "noul" else "score"])


def make_router(scenario: Scenario):
    """-> (the router for the scenario's CLM backend, a line saying what it is using)."""
    if scenario.clm_backend == "server":
        key = os.environ.get(scenario.clm_key_env or "", "") or None
        note = f"the CLM server at {scenario.clm_url}, model {scenario.clm_model_id}"
        if scenario.clm_heads:
            note += " (trained heads apply to the mlx backend only, so the heads setting is ignored)"
        return LocalContrastiveRouter(engine=RemoteEngine(scenario.clm_url, scenario.clm_model_id, key)), note
    heads_dir, note = resolve_heads(scenario)
    return LocalContrastiveRouter(scenario.clm_model_id, heads_dir=heads_dir), f"its {note}"


def last_turn(state_text: str) -> str:
    """The final user message of a rendered conversation, with any continuation lines."""
    lines = state_text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith("User: ")]
    return "\n".join(lines[starts[-1]:]) if starts else state_text


class LocalContrastiveRouter:
    """CLM for one scenario: routing decisions, tool-call checks and reuse of reviewed routes."""

    def __init__(self, model_id: str = DEFAULT_CLM, engine=None, heads_dir=None):
        self.engine = engine if engine is not None else load_engine(model_id, heads_dir)
        # A server gives no vectors to compare, so reviewed routes are then matched on exact text
        self.exact_match = isinstance(self.engine, RemoteEngine)
        self.verdicts: list = []

    def decide(self, state_text: str, scenario: Scenario) -> dict:
        """One CLM call: the routing question plus the scenario's extra questions."""
        questions = {
            ROUTE_QID: {
                "type": "choice",
                "instructions": scenario.routing_instructions,
                "criteria": {name: route.description for name, route in scenario.routes.items()},
            },
            **scenario.questions,
        }
        answers = self.engine.answer(state_text, questions)["answers"]
        route = answers.pop(ROUTE_QID)
        decision = {
            "route": route["choice"],
            "confidence": route["confidence"],
            "probabilities": route["probabilities"],
            "answers": answers,
            "source": "clm",
        }
        if self.verdicts and scenario.live_corrections:
            match = self._reviewed(state_text, scenario)
            latest = last_turn(state_text)
            if not match and latest != state_text:
                # A question repeated later in a conversation should still meet its first-turn verdict
                match = self._reviewed(latest, scenario)
            if match:
                human_route, similarity = match
                decision.update(route=human_route, source="reviewed", similarity=similarity,
                                clm_route=route["choice"])
        return decision

    def _reviewed(self, state_text: str, scenario: Scenario) -> tuple[str, float] | None:
        """The route a person chose for this state before, if one matches: (route, similarity)."""
        if self.exact_match:
            routes = [route for text, route in self.verdicts if text == state_text]
            return (routes[-1], 1.0) if routes else None
        return self.match_verdict(self.embed_state(state_text, scenario), scenario.match_threshold)

    def embed_state(self, state_text: str, scenario: Scenario) -> list[float]:
        """CLM's state-head vector for this state under the routing question (unit length)."""
        # Same layout as clm_mlx.schema.state_text: context, blank line, question
        text = f"{state_text.strip()}\n\n{scenario.routing_instructions.strip()}"
        embedding, _ = self.engine.encoder.embed([text])
        return self.engine.heads.project(embedding, "state")[0].tolist()

    def load_verdicts(self, records: list[dict], scenario: Scenario) -> int:
        """Embed the reviewed routes in `records` for matching in decide(); returns how many."""
        self.verdicts = [(r["state_text"] if self.exact_match else self.embed_state(r["state_text"], scenario),
                          r["human"]) for r in route_verdicts(records, scenario)]
        return len(self.verdicts)

    def match_verdict(self, vector: list[float], threshold: float) -> tuple[str, float] | None:
        """The closest stored verdict at or above the threshold; on a tie the later one wins."""
        best_route, best_similarity = None, threshold
        for stored, human_route in self.verdicts:
            similarity = sum(a * b for a, b in zip(vector, stored))
            if similarity >= best_similarity:
                best_route, best_similarity = human_route, similarity
        return None if best_route is None else (best_route, best_similarity)

    def check_tool(self, state_text: str, scenario: Scenario) -> float:
        """Probability that a proposed tool call is appropriate; the call is described in state_text."""
        question = {"type": "noul", "instructions": scenario.tool_check_instructions}
        return self.engine.answer(state_text, {"tool": question})["answers"]["tool"]["noul"]

    def compute_alignment(self, state: str, action_palette: dict[str, str],
                          instructions: str) -> list[tuple[str, float]]:
        """
        CLM inference: the state head embeds `state + instructions`, the action head
        embeds each option's description, and the answer is a softmax over the
        scaled cosine similarities. No text is generated.
        """
        question = {"type": "choice", "instructions": instructions, "criteria": action_palette}
        answer = self.engine.answer(state, {"route": question})["answers"]["route"]
        return sorted(answer["probabilities"].items(), key=lambda x: x[1], reverse=True)

    def ask(self, state: str, questions: dict[str, dict]) -> dict:
        """Typed questions (noul / choice / score) answered against one state in a single pass."""
        return self.engine.answer(state, questions)["answers"]
