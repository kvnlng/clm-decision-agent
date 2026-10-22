"""Labelled example messages for a scenario: where they live and how they are split for training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import clm_schema
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


def labels(scenario: Scenario, example: dict) -> dict[str, int]:
    """The line's answers as candidate indexes, {question: index}: true or false for a yes/no question, the
    level's index for a score, the key for a choice. Raises ValueError saying what is wrong."""
    answers = example.get("answers", {})
    if not isinstance(answers, dict):
        raise ValueError('"answers" must be an object of question names and answers')
    found = {}
    for name, value in answers.items():
        if name not in scenario.questions:
            raise ValueError(f"unknown question '{name}'")
        question = scenario.questions[name]
        keys, _ = clm_schema.candidates(question)
        where = f"questions.{name}"
        if question["type"] == "noul":
            if not isinstance(value, bool):
                raise ValueError(f"{where}: needs true or false, not {value!r}")
            found[name] = keys.index("true" if value else "false")
        elif question["type"] == "score":
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(keys):
                raise ValueError(f"{where}: {value!r} is not a level (0 to {len(keys) - 1})")
            found[name] = value
        else:
            if value not in keys:
                raise ValueError(f"{where}: {value!r} is not one of {', '.join(keys)}")
            found[name] = keys.index(value)
    return found


def _row(scenario: Scenario, example: dict, text: str | None = None) -> dict:
    """A training row from an examples line or a correction: the text, route, answers, facts and options."""
    row = {"text": text if text is not None else example["text"], "route": example["route"],
           "answers": labels(scenario, example), "facts": list(example.get("facts", [])), "options": None,
           "option": None}
    if "options" in example or "option" in example:
        if scenario.routes[example["route"]].options is None:
            raise ValueError(f"routes.{example['route']} has no options to choose among")
        options = example.get("options")
        if not (isinstance(options, list) and all(isinstance(o, dict) and isinstance(o.get("id"), str)
                                                  and isinstance(o.get("text"), str) for o in options)):
            raise ValueError('"options" must be a list of {"id": ..., "text": ...}')
        ids = [option["id"] for option in options]
        if example.get("option") not in ids:
            raise ValueError(f"option {example.get('option')!r} is not among the line's options")
        row.update(options=options, option=ids.index(example["option"]))
    return row


def training_rows(scenario: Scenario, examples: list[dict], records: list[dict]) -> tuple[list, list, list]:
    """-> (train, reviewed, test) lists of rows: {"text", "route", "answers" ({question: candidate index}),
    "facts", "options" (or None), "option" (the right option's index, or None)}.

    `reviewed` are the routes the owner chose in review mode. They are real, so all of them are trained on,
    except one that repeats a held-back example. Raises ValueError naming the examples line that is wrong."""
    train, test = [], []
    for number, example in enumerate(examples, start=1):
        if example["route"] in scenario.routes:
            try:
                row = _row(scenario, example)
            except ValueError as error:
                raise ValueError(f"line {number}: {error}") from error
            (test if split_of(example["text"]) == "test" else train).append(row)
    held_back = {row["text"] for row in test}
    # A reviewed route's text may carry routing's facts after the message ("...\n\nKnown:\n- ..."); the
    # message is what repeats an example
    reviewed = [{"text": verdict["state_text"], "route": verdict["human"], "answers": {}, "facts": [],
                 "options": None, "option": None}
                for verdict in route_verdicts(records, scenario)
                if verdict["state_text"].split("\n\nKnown:\n")[0] not in held_back]
    return train, reviewed, test
