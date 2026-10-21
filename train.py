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
import json
import math
import random
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten

from clm_router import HEADS_FILE, load_engine
from dataset import examples_path, load_examples, training_rows
from record import load_records, record_key
from scenario import ScenarioError
from scenario import load_for_command


def _unit(x: mx.array) -> mx.array:
    return x / (mx.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class HeadsModel(nn.Module):
    """The state head and the action head, scored as CLM scores them: scaled cosine similarity."""

    def __init__(self, state_head: nn.Module, action_head: nn.Module, scale: float):
        super().__init__()
        self.state_head, self.action_head, self.scale = state_head, action_head, scale

    def logits(self, states: mx.array, actions: mx.array) -> mx.array:
        """[n states, k options] scores from encoder vectors of the states and of the options."""
        return self.scale * (_unit(self.state_head(_unit(states))) @ _unit(self.action_head(_unit(actions))).T)


def accuracy(model: HeadsModel, states: mx.array, labels: mx.array, actions: mx.array) -> float:
    """The share of states whose highest-scoring option is the labelled one."""
    return float(mx.mean(mx.argmax(model.logits(states, actions), axis=1) == labels))


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def fit(model: HeadsModel, states: mx.array, labels: mx.array, actions: mx.array, epochs: int = 25,
        learning_rate: float = 5e-5, batch: int = 32, patience: int = 4, seed: int = 0, pinned: int = 0) -> dict:
    """Train, and keep the weights that did best on a fifth of the examples set aside for validation.

    The last `pinned` rows are never set aside. The starting weights count as a candidate, so
    training that only does harm changes nothing."""
    order = list(range(len(labels) - pinned))
    random.Random(seed).shuffle(order)
    held = max(1, len(order) // 5)
    validation = mx.array(order[:held])
    training = order[held:] + list(range(len(labels) - pinned, len(labels)))
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-2)

    def loss(current, rows):
        return mx.mean(nn.losses.cross_entropy(current.logits(states[rows], actions), labels[rows]))
    step = nn.value_and_grad(model, loss)

    def snapshot():
        return [(name, mx.array(value)) for name, value in tree_flatten(model.parameters())]

    best = accuracy(model, states[validation], labels[validation], actions)
    best_weights, best_epoch, epoch = snapshot(), 0, 0
    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(training)
        for start in range(0, len(training), batch):
            _, gradients = step(model, mx.array(training[start:start + batch]))
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
        score = accuracy(model, states[validation], labels[validation], actions)
        if score > best + 1e-6:
            best, best_weights, best_epoch = score, snapshot(), epoch
        elif epoch - best_epoch >= patience:
            break
    model.load_weights(best_weights)
    mx.eval(model.parameters())
    return {"epochs": epoch, "best_epoch": best_epoch, "validation_accuracy": best, "validation_examples": held}


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def probe_accuracy(train_states, train_labels, test_states, test_labels, classes: int, steps: int = 400) -> float:
    """A baseline: a plain linear classifier on the encoder's vectors, with no CLM heads involved."""
    mx.random.seed(0)
    linear = nn.Linear(train_states.shape[1], classes)
    inputs = _unit(train_states) * 30.0
    optimizer = optim.AdamW(learning_rate=3e-3, weight_decay=1e-2)
    step = nn.value_and_grad(linear, lambda m: mx.mean(nn.losses.cross_entropy(m(inputs), train_labels)))
    for _ in range(steps):
        _, gradients = step(linear)
        optimizer.update(linear, gradients)
        mx.eval(linear.parameters(), optimizer.state)
    return float(mx.mean(mx.argmax(linear(_unit(test_states) * 30.0), axis=1) == test_labels))


def save_heads(model: HeadsModel, released_dir, out_dir, facts: dict) -> None:
    """Write the heads in the released layout, plus training.json describing how they were made."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # training.json is what marks the folder as usable, so it goes first and comes back last:
    # an interrupted save then leaves heads the server will not try to load
    (out_dir / "training.json").unlink(missing_ok=True)
    weights = {"logit_scale": mx.array([math.log(model.scale)])}
    for prefix, head in (("state_head", model.state_head), ("action_head", model.action_head)):
        weights.update({f"{prefix}.{name}": value for name, value in tree_flatten(head.parameters())})
    mx.save_safetensors(str(out_dir / HEADS_FILE), weights)
    shutil.copy(Path(released_dir) / "config.json", out_dir / "config.json")
    (out_dir / "training.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


def _embed(engine, texts: list[str], batch: int = 32) -> mx.array:
    parts = []
    for start in range(0, len(texts), batch):
        vectors, _ = engine.encoder.embed(texts[start:start + batch])
        mx.eval(vectors)
        parts.append(vectors)
    return mx.concatenate(parts)


def _other_answers(engine, scenario, states: list[str]) -> list[dict]:
    engine.cache.clear()
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
        return _embed(engine, texts), mx.array([names.index(route) for _, route in rows])

    write(f"Encoding {len(train_rows)} examples and {len(reviewed)} reviewed routes for training, "
          f"and {len(test_rows)} held-back examples...")
    (train_states, train_labels), (test_states, test_labels) = encode(train_rows + reviewed), encode(test_rows)
    actions = _embed(engine, [route.description for route in scenario.routes.values()])
    model = HeadsModel(engine.heads.state, engine.heads.action, engine.heads.scale)
    sample = [state for state, _ in test_rows[::max(1, len(test_rows) // 30)]]       # spread across the routes

    before, others_before = accuracy(model, test_states, test_labels, actions), _other_answers(engine, scenario, sample)
    info = fit(model, train_states, train_labels, actions, pinned=len(reviewed))
    after, others_after = accuracy(model, test_states, test_labels, actions), _other_answers(engine, scenario, sample)
    engine.cache.clear()
    baseline = probe_accuracy(train_states, train_labels, test_states, test_labels, len(names))

    write(f"Routing accuracy on {len(test_rows)} held-back generated examples:")
    write(f"  released heads      {before:.0%}")
    write(f"  trained heads       {after:.0%}   "
          f"(kept after pass {info['best_epoch']} of {info['epochs']} over the training examples)")
    write(f"  simple classifier   {baseline:.0%}   (a baseline on the encoder's vectors; not used by the server)")
    correct = mx.argmax(model.logits(test_states, actions), axis=1) == test_labels
    for index, name in enumerate(names):
        mine = test_labels == index
        total = int(mx.sum(mine))
        if total:
            write(f"    {name:<24}{int(mx.sum(correct & mine)):>4} / {total}")
    drift = _drift(others_before, others_after)
    for name, change in drift.items():
        write(f"  untrained question '{name}': {change}")
    return {"routing_sha": scenario.routing_sha, "clm_model_id": scenario.clm_model_id,
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
    if scenario.clm_backend != "mlx":
        print("error: train.py fine-tunes the heads of the mlx backend. With a CLM server, train on the server "
              "with the CLM authors' tools.", file=sys.stderr)
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
    print(f"Loading CLM ({scenario.clm_model_id})...")
    engine = load_engine(scenario.clm_model_id)
    facts = train(scenario, engine, train_rows, reviewed, test_rows)
    if facts["accuracy_after"] <= facts["accuracy_before"]:
        print("Training did not improve routing on the held-back examples, so nothing was saved.")
        return 1
    released = Path(sys.modules["clm_mlx"].__file__).parent.parent / "heads"
    save_heads(HeadsModel(engine.heads.state, engine.heads.action, engine.heads.scale), released,
               scenario.clm_heads, facts)
    print(f"Trained heads saved to {scenario.clm_heads}. The server uses them from its next start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
