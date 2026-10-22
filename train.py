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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import nn

import clm_schema
from clm_router import HEADS_FILE, ClmError, load_engine
from clm_torch import Head
from dataset import examples_path, load_examples, training_rows
from decision import routed_text
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
        def head(prefix: str) -> Head:
            made = Head.from_config(config)
            made.load_state_dict({name[len(prefix) + 1:]: torch.as_tensor(np.asarray(value, dtype=np.float32))
                                  for name, value in weights.items() if name.startswith(prefix + ".")}, strict=True)
            return made
        scale = min(math.exp(float(np.asarray(weights["logit_scale"]).reshape(-1)[0])), 100.0)
        return cls(head("state_head"), head("action_head"), scale)

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


class MlxAdamW:
    """AdamW as MLX applies it, which the example's heads were first trained with: weight decay first, and
    no bias correction, so early steps are larger than torch's AdamW takes."""

    # pylint: disable-next=too-many-arguments
    def __init__(self, parameters, learning_rate: float, weight_decay: float = 1e-2, betas=(0.9, 0.999),
                 eps: float = 1e-8):
        self.parameters = list(parameters)
        self.learning_rate, self.weight_decay, self.betas, self.eps = learning_rate, weight_decay, betas, eps
        self.state = [(torch.zeros_like(p), torch.zeros_like(p)) for p in self.parameters]

    def zero_grad(self) -> None:
        """Forget the last gradients."""
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        """Move each parameter by its gradient."""
        b1, b2 = self.betas
        for parameter, (m, v) in zip(self.parameters, self.state):
            if parameter.grad is None:
                continue
            parameter.mul_(1 - self.learning_rate * self.weight_decay)
            m.mul_(b1).add_(parameter.grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(parameter.grad, parameter.grad, value=1 - b2)
            parameter.addcdiv_(m, v.sqrt().add_(self.eps), value=-self.learning_rate)


def accuracy(model: HeadsModel, states: torch.Tensor, labels: torch.Tensor, actions: torch.Tensor) -> float:
    """The share of states whose highest-scoring option is the labelled one."""
    with torch.no_grad():
        return float((model.logits(states, actions).argmax(dim=1) == labels).float().mean())


@dataclass
class Task:
    """One thing CLM is trained to answer: the rows labelled for it, their state vectors, the right candidate of
    each, and the candidates. A question shares its candidates (`classes` names them); ranking has its own
    candidates per row (`actions` is then a list, and `classes` None)."""

    name: str
    rows: torch.Tensor                          # [m] which rows carry this label
    states: torch.Tensor                        # [m, hidden]
    labels: torch.Tensor                        # [m] the right candidate's index
    actions: torch.Tensor | list[torch.Tensor]  # [k, hidden], or one [k_i, hidden] per row
    classes: list[str] | None = None
    weights: torch.Tensor | None = None         # per class, so a rare answer counts as much as a common one

    def logits(self, model: HeadsModel, entries: torch.Tensor) -> list[torch.Tensor] | torch.Tensor:
        """The scores of the entries' candidates: [n, k] for a question, a list of [k_i] for ranking."""
        if self.classes is not None:
            return model.logits(self.states[entries], self.actions)
        return [model.logits(self.states[i:i + 1], self.actions[i])[0] for i in entries.tolist()]

    def loss(self, model: HeadsModel, entries: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over the entries, weighted by class for a question."""
        scores = self.logits(model, entries)
        if self.classes is not None:
            return nn.functional.cross_entropy(scores, self.labels[entries], weight=self.weights)
        return torch.stack([nn.functional.cross_entropy(s[None], self.labels[i:i + 1])
                            for s, i in zip(scores, entries.tolist())]).mean()

    def predictions(self, model: HeadsModel, entries: torch.Tensor) -> torch.Tensor:
        """The highest-scoring candidate of each entry."""
        with torch.no_grad():
            scores = self.logits(model, entries)
            if self.classes is not None:
                return scores.argmax(dim=1)
            return torch.tensor([int(s.argmax()) for s in scores], dtype=torch.long)

    def per_class(self, model: HeadsModel, entries: torch.Tensor | None = None) -> dict[str, list[int]]:
        """{class: [right, of]} over the entries (all of them by default); ranking has the one class
        "right option"."""
        entries = torch.arange(len(self.labels)) if entries is None else entries
        right = self.predictions(model, entries) == self.labels[entries]
        if self.classes is None:
            return {"right option": [int(right.sum()), len(entries)]}
        wanted = self.labels[entries]
        return {name: [int((right & (wanted == k)).sum()), int((wanted == k).sum())]
                for k, name in enumerate(self.classes) if int((wanted == k).sum())}

    def balanced(self, model: HeadsModel, entries: torch.Tensor | None = None) -> float:
        """The mean of each class's share answered right, so "always the common answer" scores low."""
        found = self.per_class(model, entries)
        return sum(right / total for right, total in found.values()) / len(found) if found else 0.0


def class_weights(labels: torch.Tensor, classes: int) -> torch.Tensor:
    """n / (k * count) for each of the k classes, and 0 for a class with no examples."""
    counts = torch.bincount(labels, minlength=classes).float()
    return torch.where(counts > 0, len(labels) / (classes * counts.clamp(min=1)), torch.zeros_like(counts))


def _entries(task: Task, rows: list[int]) -> torch.Tensor:
    return torch.isin(task.rows, torch.tensor(rows, dtype=torch.long)).nonzero().flatten()


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def fit_tasks(model: HeadsModel, tasks: list[Task], rows: int, epochs: int = 25, learning_rate: float = 5e-5,
              batch: int = 32, patience: int = 4, seed: int = 0, pinned: int = 0) -> dict:
    """Train on every task at once, and keep the weights that did best on a fifth of the `rows` set aside for
    validation, scored as the mean balanced accuracy over the tasks.

    The last `pinned` rows are never set aside. The starting weights count as a candidate, so
    training that only does harm changes nothing."""
    order = list(range(rows - pinned))
    random.Random(seed).shuffle(order)
    held = max(1, len(order) // 5)
    validation = [_entries(task, order[:held]) for task in tasks]
    training = order[held:] + list(range(rows - pinned, rows))
    optimizer = MlxAdamW(model.parameters(), learning_rate=learning_rate, weight_decay=1e-2)

    def score() -> float:
        found = [task.balanced(model, entries) for task, entries in zip(tasks, validation) if len(entries)]
        return sum(found) / len(found) if found else 0.0

    best = score()
    best_weights, best_epoch, epoch = copy.deepcopy(model.state_dict()), 0, 0
    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(training)
        for start in range(0, len(training), batch):
            chosen = training[start:start + batch]
            losses = [task.loss(model, entries) for task in tasks if len(entries := _entries(task, chosen))]
            if not losses:
                continue
            optimizer.zero_grad()
            sum(losses).backward()
            optimizer.step()
        found = score()
        if found > best + 1e-6:
            best, best_weights, best_epoch = found, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_weights)
    return {"epochs": epoch, "best_epoch": best_epoch, "validation_accuracy": best, "validation_examples": held}


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def fit(model: HeadsModel, states: torch.Tensor, labels: torch.Tensor, actions: torch.Tensor, epochs: int = 25,
        learning_rate: float = 5e-5, batch: int = 32, patience: int = 4, seed: int = 0, pinned: int = 0) -> dict:
    """fit_tasks for one question whose rows are all labelled: its states, labels and candidates."""
    task = Task("route", torch.arange(len(labels)), states, labels, actions,
                classes=[str(k) for k in range(len(actions))], weights=class_weights(labels, len(actions)))
    return fit_tasks(model, [task], len(labels), epochs, learning_rate, batch, patience, seed, pinned)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def probe_accuracy(train_states, train_labels, test_states, test_labels, classes: int, steps: int = 400) -> float:
    """A baseline: a plain linear classifier on the encoder's vectors, with no CLM heads involved."""
    torch.manual_seed(0)
    linear = nn.Linear(train_states.shape[1], classes)
    inputs = _unit(train_states) * 30.0
    optimizer = MlxAdamW(linear.parameters(), learning_rate=3e-3, weight_decay=1e-2)
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


def _other_answers(engine, questions: dict, states: list[str]) -> list[dict]:
    return [engine.answer(state, questions)["answers"] for state in states] if questions else []


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


def _question_text(row: dict, instructions: str) -> str:
    """The text CLM reads for a question about the row: the request, its facts, then the question."""
    # A correction's text may carry routing's facts; its own facts include them
    return clm_schema.state_text(routed_text(row.get("message", row["text"]), row["facts"]), instructions)


class _Candidates:  # pylint: disable=too-few-public-methods
    """Candidate texts' vectors, each list encoded once for training and testing alike."""

    def __init__(self, engine):
        self.engine, self.found = engine, {}

    def __call__(self, texts: list[str]) -> torch.Tensor:
        if tuple(texts) not in self.found:
            self.found[tuple(texts)] = _embed(self.engine, texts)
        return self.found[tuple(texts)]


def _question_task(name: str, question: dict, engine, rows: list[dict], candidates: _Candidates) -> Task:
    keys, texts = clm_schema.candidates(question)
    mine = [i for i, row in enumerate(rows) if name in row["answers"]]
    labels = torch.tensor([rows[i]["answers"][name] for i in mine], dtype=torch.long)
    states = _embed(engine, [_question_text(rows[i], question["instructions"]) for i in mine]) if mine \
        else torch.zeros(0)
    return Task(name, torch.tensor(mine, dtype=torch.long), states, labels, candidates(texts), classes=keys,
                weights=class_weights(labels, len(keys)) if mine else None)


def _ranking_task(route, engine, rows: list[dict], candidates: _Candidates) -> Task:
    mine = [i for i, row in enumerate(rows) if row["options"] and row["route"] == route.name]
    states = _embed(engine, [_question_text(rows[i], route.options.instructions) for i in mine]) if mine \
        else torch.zeros(0)
    return Task(f"options of {route.name}", torch.tensor(mine, dtype=torch.long), states,
                torch.tensor([rows[i]["option"] for i in mine], dtype=torch.long),
                [candidates([option["text"] for option in rows[i]["options"]]) for i in mine])


def _tasks(scenario, engine, rows: list[dict], candidates: _Candidates, trained: list[str] | None = None) -> list[Task]:
    """A task for the route, each question the rows label, and each route's ranking the rows label; with
    `trained`, the tasks of those names instead."""
    names = list(scenario.routes)
    labels = torch.tensor([names.index(row["route"]) for row in rows])
    # The same layout CLM is asked with: the request, a blank line, the routing question
    tasks = [Task("route", torch.arange(len(rows)),
                  _embed(engine, [f"{row['text'].strip()}\n\n{scenario.routing_instructions.strip()}"
                                  for row in rows]),
                  labels, candidates([route.description for route in scenario.routes.values()]), classes=names,
                  weights=class_weights(labels, len(names)))]
    for name, question in scenario.questions.items():
        if (name in trained) if trained is not None else any(name in row["answers"] for row in rows):
            tasks.append(_question_task(name, question, engine, rows, candidates))
    for route in scenario.routes.values():
        wanted = f"options of {route.name}"
        if route.options and ((wanted in trained) if trained is not None
                              else any(row["options"] and row["route"] == route.name for row in rows)):
            tasks.append(_ranking_task(route, engine, rows, candidates))
    return tasks


def _write_results(write, tasks: list[Task], before: dict, after: dict) -> None:
    for task in tasks:
        write(f"  {task.name:<24}released {before[task.name]['score']:.0%}   "
              f"trained {after[task.name]['score']:.0%}   (balanced: each class counts the same)")
        for name, (right, total) in after[task.name]["per_class"].items():
            was = before[task.name]["per_class"].get(name, [0, total])[0]
            write(f"    {name:<22}released {was:>3} of {total:<4} trained {right:>3} of {total}")


def _results(model: HeadsModel, tasks: list[Task]) -> dict:
    return {task.name: {"score": task.balanced(model), "per_class": task.per_class(model)} for task in tasks}


# pylint: disable-next=too-many-locals,too-many-arguments,too-many-positional-arguments
def train(scenario, engine, train_rows: list, reviewed: list, test_rows: list, write=print) -> dict:
    """Fine-tune the engine's heads in place on the rows: the route, every question the rows label, and each
    route's ranking of options. Returns the facts saved with the heads."""
    write(f"Encoding {len(train_rows)} examples and {len(reviewed)} reviewed routes for training, "
          f"and {len(test_rows)} held-back examples...")
    candidates = _Candidates(engine)
    tasks = _tasks(scenario, engine, train_rows + reviewed, candidates)
    trained = [task.name for task in tasks]
    test_tasks = _tasks(scenario, engine, test_rows, candidates, trained)
    model = HeadsModel.from_weights(engine.head_weights(), engine.head_config)
    sample = [row["text"] for row in test_rows[::max(1, len(test_rows) // 30)]]       # spread across the routes
    untrained = {name: question for name, question in scenario.questions.items() if name not in trained}
    others_before = _other_answers(engine, untrained, sample)

    before = _results(model, test_tasks)
    info = fit_tasks(model, tasks, len(train_rows) + len(reviewed), pinned=len(reviewed))
    engine.set_head_weights(model.weights())
    after = _results(model, test_tasks)
    route, test_route = tasks[0], test_tasks[0]
    baseline = probe_accuracy(route.states, route.labels, test_route.states, test_route.labels, len(scenario.routes))

    write(f"Results on {len(test_rows)} held-back generated examples "
          f"(kept after pass {info['best_epoch']} of {info['epochs']} over the training examples):")
    _write_results(write, test_tasks, before, after)
    write(f"  simple classifier for the route {baseline:.0%}   (a baseline on the encoder's vectors; "
          f"not used by the server)")
    drift = _drift(others_before, _other_answers(engine, untrained, sample))
    for name, change in drift.items():
        write(f"  untrained question '{name}': {change}")
    score_before = sum(found["score"] for found in before.values()) / len(before)
    score_after = sum(found["score"] for found in after.values()) / len(after)
    return {"routing_sha": scenario.routing_sha, "clm_backend": scenario.clm_backend,
            "clm_model_id": scenario.clm_model_id,
            "train_examples": len(train_rows), "reviewed_routes": len(reviewed), "test_examples": len(test_rows),
            "trained": trained,
            "questions_sha": {name: scenario.question_sha(name) for name in trained if name != "route"},
            "options_trained": [name[len("options of "):] for name in trained
                                                    if name.startswith("options of ")],
            "accuracy_before": _plain(before["route"]["per_class"]),
            "accuracy_after": _plain(after["route"]["per_class"]),
            "score_before": score_before, "score_after": score_after, "accuracy_simple_classifier": baseline,
            "per_class": {name: {"before": before[name]["per_class"], "after": after[name]["per_class"]}
                          for name in trained},
            "other_questions": drift, **info}


def _plain(per_class: dict) -> float:
    """The share answered right, all classes together."""
    total = sum(of for _, of in per_class.values())
    return sum(right for right, _ in per_class.values()) / total if total else 0.0


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
    except ValueError as error:
        print(f"error: {examples_path(scenario).name}: {error}", file=sys.stderr)
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
    if facts["score_after"] <= facts["score_before"]:
        print("Training did not improve the trained answers on the held-back examples, so nothing was saved.")
        return 1
    save_heads(engine.head_weights(), engine.head_config, scenario.clm_heads, facts)
    print(f"Trained heads saved to {scenario.clm_heads}. The server uses them from its next start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
