"""Fine-tune CLM's two heads on a scenario's labelled examples, so routing fits the scenario.

    uv run python train.py <scenario>

The encoder stays frozen. Each example's state (the message plus the routing question) and each
route's description are encoded once; training then only adjusts the two small heads so that a
state lands nearest the description of its route. It takes seconds.
"""
import os
# Set before huggingface_hub is imported, which reads it at import time
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# pylint: disable=wrong-import-position

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import nn

from clm_router import HEADS_FILE, ClmError, load_engine
from clm_torch import Head
from dataset import examples_path, load_examples, training_rows
from record import load_records, record_key
from scenario import ScenarioError
from scenario import load_for_command


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


class HeadsModel(nn.Module):  # pylint: disable=abstract-method
    """The state head and the action head, scored as CLM scores them: scaled cosine similarity."""

    def __init__(self, state_head: nn.Module, action_head: nn.Module, scale: float):
        super().__init__()
        self.state_head, self.action_head, self.scale = state_head, action_head, scale

    @classmethod
    def from_weights(cls, weights: dict[str, np.ndarray], config: dict) -> "HeadsModel":
        """The heads an engine is using, from its head_weights() and head_config."""
        heads = []
        for prefix in ("state_head", "action_head"):
            head = Head.from_config(config)
            head.load_state_dict({name[len(prefix) + 1:]: torch.as_tensor(np.asarray(value, dtype=np.float32))
                                  for name, value in weights.items() if name.startswith(prefix + ".")}, strict=True)
            heads.append(head)
        state_head, action_head = heads
        return cls(state_head, action_head, min(math.exp(float(np.asarray(weights["logit_scale"]).reshape(-1)[0])),
                                                100.0))

    def weights(self) -> dict[str, np.ndarray]:
        """The weights in the layout of a heads file."""
        weights = {"logit_scale": np.array([math.log(self.scale)], dtype=np.float32)}
        for prefix, head in (("state_head", self.state_head), ("action_head", self.action_head)):
            weights.update({f"{prefix}.{name}": value.detach().cpu().numpy().copy()
                            for name, value in head.state_dict().items()})
        return weights

    def logits(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """[n states, k options] scores from encoder vectors of the states and of the options."""
        return self.scale * (_unit(self.state_head(_unit(states))) @ _unit(self.action_head(_unit(actions))).T)


def accuracy(model: HeadsModel, states: torch.Tensor, labels: torch.Tensor, actions: torch.Tensor) -> float:
    """The share of states whose highest-scoring option is the labelled one."""
    with torch.no_grad():
        return float((model.logits(states, actions).argmax(dim=1) == labels).float().mean())


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def fit(model: HeadsModel, states: torch.Tensor, labels: torch.Tensor, actions: torch.Tensor, epochs: int = 25,
        learning_rate: float = 5e-5, batch: int = 32, patience: int = 4, seed: int = 0, pinned: int = 0) -> dict:
    """Train, and keep the weights that did best on a fifth of the examples set aside for validation.

    The last `pinned` rows are never set aside. The starting weights count as a candidate, so
    training that only does harm changes nothing."""
    order = list(range(len(labels) - pinned))
    random.Random(seed).shuffle(order)
    held = max(1, len(order) // 5)
    validation = torch.tensor(order[:held])
    training = order[held:] + list(range(len(labels) - pinned, len(labels)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    best = accuracy(model, states[validation], labels[validation], actions)
    best_weights, best_epoch, epoch = copy.deepcopy(model.state_dict()), 0, 0
    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(training)
        for start in range(0, len(training), batch):
            rows = torch.tensor(training[start:start + batch])
            optimizer.zero_grad()
            nn.functional.cross_entropy(model.logits(states[rows], actions), labels[rows]).backward()
            optimizer.step()
        score = accuracy(model, states[validation], labels[validation], actions)
        if score > best + 1e-6:
            best, best_weights, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_weights)
    return {"epochs": epoch, "best_epoch": best_epoch, "validation_accuracy": best, "validation_examples": held}


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def probe_accuracy(train_states, train_labels, test_states, test_labels, classes: int, steps: int = 400) -> float:
    """A baseline: a plain linear classifier on the encoder's vectors, with no CLM heads involved."""
    torch.manual_seed(0)
    linear = nn.Linear(train_states.shape[1], classes)
    inputs = _unit(train_states) * 30.0
    optimizer = torch.optim.AdamW(linear.parameters(), lr=3e-3, weight_decay=1e-2)
    for _ in range(steps):
        optimizer.zero_grad()
        nn.functional.cross_entropy(linear(inputs), train_labels).backward()
        optimizer.step()
    with torch.no_grad():
        return float((linear(_unit(test_states) * 30.0).argmax(dim=1) == test_labels).float().mean())


def save_heads(weights: dict[str, np.ndarray], config: dict, out_dir, facts: dict) -> None:
    """Write the heads in the released layout, with their config.json and training.json describing how they
    were made. Both backends load this layout."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # training.json is what marks the folder as usable, so it goes first and comes back last:
    # an interrupted save then leaves heads the server will not try to load
    (out_dir / "training.json").unlink(missing_ok=True)
    save_file({name: np.ascontiguousarray(value, dtype=np.float32) for name, value in weights.items()},
              str(out_dir / HEADS_FILE))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "training.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


def _embed(engine, texts: list[str], batch: int = 32) -> torch.Tensor:
    return torch.as_tensor(np.concatenate([engine.embed(texts[start:start + batch])
                                           for start in range(0, len(texts), batch)]))


def _other_answers(engine, scenario, states: list[str]) -> list[dict]:
    return [engine.answer(state, scenario.questions)["answers"] for state in states] if scenario.questions else []


def _drift(before: list[dict], after: list[dict]) -> dict[str, str]:
    """How far training moved the answers to the scenario's other questions, which it did not train."""
    report = {}
    for name in (before[0] if before else {}):
        pairs = [(b[name], a[name]) for b, a in zip(before, after)]
        if pairs[0][0]["type"] == "choice":
            changed = sum(b["choice"] != a["choice"] for b, a in pairs)
            report[name] = f"{changed} of {len(pairs)} answers changed"
        else:
            key = "noul" if pairs[0][0]["type"] == "noul" else "score"
            report[name] = f"moved by {sum(abs(b[key] - a[key]) for b, a in pairs) / len(pairs):.2f} on average"
    return report


# pylint: disable-next=too-many-locals,too-many-arguments,too-many-positional-arguments
def train(scenario, engine, train_rows: list, reviewed: list, test_rows: list, write=print) -> dict:
    """Fine-tune the engine's heads in place on the rows; returns the facts saved with the heads."""
    names = list(scenario.routes)

    def encode(rows):
        # The same layout CLM is asked with: the conversation, a blank line, the routing question
        texts = [f"{state.strip()}\n\n{scenario.routing_instructions.strip()}" for state, _ in rows]
        return _embed(engine, texts), torch.tensor([names.index(route) for _, route in rows])

    write(f"Encoding {len(train_rows)} examples and {len(reviewed)} reviewed routes for training, "
          f"and {len(test_rows)} held-back examples...")
    (train_states, train_labels), (test_states, test_labels) = encode(train_rows + reviewed), encode(test_rows)
    actions = _embed(engine, [route.description for route in scenario.routes.values()])
    model = HeadsModel.from_weights(engine.head_weights(), engine.head_config)
    sample = [state for state, _ in test_rows[::max(1, len(test_rows) // 30)]]       # spread across the routes

    before, others_before = accuracy(model, test_states, test_labels, actions), _other_answers(engine, scenario, sample)
    info = fit(model, train_states, train_labels, actions, pinned=len(reviewed))
    engine.set_head_weights(model.weights())
    after, others_after = accuracy(model, test_states, test_labels, actions), _other_answers(engine, scenario, sample)
    baseline = probe_accuracy(train_states, train_labels, test_states, test_labels, len(names))

    write(f"Routing accuracy on {len(test_rows)} held-back generated examples:")
    write(f"  released heads      {before:.0%}")
    write(f"  trained heads       {after:.0%}   "
          f"(kept after pass {info['best_epoch']} of {info['epochs']} over the training examples)")
    write(f"  simple classifier   {baseline:.0%}   (a baseline on the encoder's vectors; not used by the server)")
    with torch.no_grad():
        correct = model.logits(test_states, actions).argmax(dim=1) == test_labels
    for index, name in enumerate(names):
        mine = test_labels == index
        total = int(mine.sum())
        if total:
            write(f"    {name:<24}{int((correct & mine).sum()):>4} / {total}")
    drift = _drift(others_before, others_after)
    for name, change in drift.items():
        write(f"  untrained question '{name}': {change}")
    return {"routing_sha": scenario.routing_sha, "clm_backend": scenario.clm_backend,
            "clm_model_id": scenario.clm_model_id,
            "train_examples": len(train_rows), "reviewed_routes": len(reviewed), "test_examples": len(test_rows),
            "accuracy_before": before, "accuracy_after": after, "accuracy_simple_classifier": baseline,
            "other_questions": drift, **info}


def main(argv: list[str] | None = None) -> int:  # pylint: disable=too-many-return-statements
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Fine-tune CLM's heads on a scenario's examples.")
    parser.add_argument("scenario", help="path to the scenario file")
    args = parser.parse_args(argv)
    scenario = load_for_command(args.scenario)
    if scenario is None:
        return 2
    if scenario.clm_heads is None:
        print('error: the scenario does not say where to save the heads. Add heads = "<folder>" under [clm].',
              file=sys.stderr)
        return 2
    try:
        examples = load_examples(examples_path(scenario))
    except ValueError as error:
        print(f"error: {examples_path(scenario).name} has a line that is not JSON: {error}", file=sys.stderr)
        return 2
    try:
        records, _ = load_records(scenario.record_path, key=record_key(scenario))
    except (ScenarioError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        train_rows, reviewed, test_rows = training_rows(scenario, examples, records)
    except (KeyError, TypeError) as error:
        print(f"error: {examples_path(scenario).name}: every line needs \"text\" and \"route\" ({error!r})",
              file=sys.stderr)
        return 2
    if len(train_rows) < 20 or not test_rows:
        print(f"error: too few examples to train on ({len(train_rows)} for training, {len(test_rows)} held back). "
              f"Run `python synthesize.py {args.scenario}` first.", file=sys.stderr)
        return 2
    print(f"Loading CLM ({scenario.clm_model_id}, {scenario.clm_backend} backend)...")
    try:
        # Training starts from the released heads, never from earlier trained ones
        engine = load_engine(scenario.clm_backend, scenario.clm_model_id, None, scenario.clm_device)
    except ClmError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    facts = train(scenario, engine, train_rows, reviewed, test_rows)
    if facts["accuracy_after"] <= facts["accuracy_before"]:
        print("Training did not improve routing on the held-back examples, so nothing was saved.")
        return 1
    save_heads(engine.head_weights(), engine.head_config, scenario.clm_heads, facts)
    print(f"Trained heads saved to {scenario.clm_heads}. The server uses them from its next start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
