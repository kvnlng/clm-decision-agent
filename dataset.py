"""Labelled example messages for a scenario: where they live and how they are split for training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from record import route_verdicts
from scenario import Scenario


def examples_path(scenario: Scenario) -> Path:
    """`<scenario stem>.examples.jsonl` beside the scenario file."""
    return scenario.path.with_name(scenario.path.stem + ".examples.jsonl")


def load_examples(path: Path) -> list[dict]:
    """The examples in a file, one JSON object per line; a missing file has none."""
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def save_examples(path: Path, rows: list[dict]) -> None:
    """Write the examples, replacing the file."""
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_of(text: str) -> str:
    """"test" for about one example in five, chosen by a hash of its text so every run agrees."""
    return "test" if hashlib.sha256(text.encode("utf-8")).digest()[0] % 5 == 0 else "train"


def training_rows(scenario: Scenario, examples: list[dict], records: list[dict]) -> tuple[list, list, list]:
    """-> (train, reviewed, test) lists of (state_text, route).

    `reviewed` are the routes the owner chose in review mode. They are real, so all of them are trained on,
    except one that repeats a held-back example."""
    train, test = [], []
    for example in examples:
        if example["route"] in scenario.routes:
            row = (example["text"], example["route"])
            (test if split_of(example["text"]) == "test" else train).append(row)
    held_back = {state for state, _ in test}
    # A reviewed route's text may carry routing's facts after the message ("...\n\nKnown:\n- ..."); the
    # message is what repeats an example
    reviewed = [(verdict["state_text"], verdict["human"]) for verdict in route_verdicts(records, scenario)
                if verdict["state_text"].split("\n\nKnown:\n")[0] not in held_back]
    return train, reviewed, test
