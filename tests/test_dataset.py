"""Tests for dataset.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,unused-argument
import re

import pytest
from conftest import BASE_TOML

from dataset import examples_path, labels, load_examples, save_examples, split_of, training_rows
from scenario import load_scenario


def test_examples_live_beside_the_scenario(scenario):
    assert examples_path(scenario) == scenario.path.with_name("scenario.examples.jsonl")


def test_examples_round_trip_and_a_missing_file_is_empty(tmp_path):
    path = tmp_path / "x.examples.jsonl"
    assert load_examples(path) == []
    rows = [{"text": "héllo", "route": "Billing", "style": "terse"}]
    save_examples(path, rows)
    assert load_examples(path) == rows


def test_split_is_stable_and_holds_back_about_a_fifth():
    texts = [f"message number {i}" for i in range(500)]
    assert [split_of(t) for t in texts] == [split_of(t) for t in texts]
    held = sum(split_of(t) == "test" for t in texts)
    assert 70 <= held <= 130


def test_training_rows_use_examples_and_reviewed_routes(scenario):
    texts = [f"message number {i}" for i in range(40)]
    examples = [{"text": t, "route": "Billing"} for t in texts] + [{"text": "gone", "route": "Removed_Route"}]
    records = [{"kind": "route", "state_text": "my app crashes", "human": "Tech",
                "clm": {"route": "Billing", "confidence": 0.8, "probabilities": {}}}]
    train, reviewed, test = training_rows(scenario, examples, records)

    assert [(row["text"], row["route"]) for row in reviewed] == [("my app crashes", "Tech")]
    train = train + reviewed
    assert all(row["route"] in scenario.routes for row in train + test)
    assert {row["text"] for row in train + test} == set(texts) | {"my app crashes"}
    assert {row["text"] for row in test} == {t for t in texts if split_of(t) == "test"}
    assert not {row["text"] for row in train} & {row["text"] for row in test}


def test_a_reviewed_route_that_repeats_a_held_back_example_is_not_trained_on(scenario):
    texts = [f"message number {i}" for i in range(40)]
    held = next(t for t in texts if split_of(t) == "test")
    records = [{"kind": "route", "state_text": held, "human": "Billing",
                "clm": {"route": "Tech", "confidence": 0.8, "probabilities": {}}}]
    _, reviewed, _ = training_rows(scenario, [{"text": t, "route": "Billing"} for t in texts], records)
    assert reviewed == []


def test_a_reviewed_route_with_routing_facts_that_repeats_a_held_back_example_is_not_trained_on(scenario):
    texts = [f"message number {i}" for i in range(40)]
    held = next(t for t in texts if split_of(t) == "test")
    records = [{"kind": "route", "state_text": f"{held}\n\nKnown:\n- The table holds personal data.",
                "human": "Billing", "clm": {"route": "Tech", "confidence": 0.8, "probabilities": {}}}]
    _, reviewed, _ = training_rows(scenario, [{"text": t, "route": "Billing"} for t in texts], records)
    assert reviewed == []


QUESTIONS = BASE_TOML.replace("[[actions]]", '''[questions.level]
type = "score"
instructions = "How soon?"
criteria = ["No deadline", "Weeks", "Days"]

[questions.team]
type = "choice"
instructions = "Which team?"
criteria = { reporting = "Dashboards", data = "Tables" }

[routes.Billing.options]
tool = "alternatives"
instructions = "Which refund suits them?"

[[actions]]''').replace('inform = ["lookup"]', 'inform = ["lookup", "alternatives"]')
REFUNDS = [{"id": "card", "text": "Back to the card"}, {"id": "credit", "text": "Store credit"}]


@pytest.fixture
def questions(write_scenario):
    return load_scenario(write_scenario(QUESTIONS))


def line(**more):
    return {"text": "I was charged twice", "route": "Billing", **more}


def test_labels_map_each_question_type_to_its_candidate_index(questions):
    # noul candidates come in the order false, true, as CLM answers them
    assert labels(questions, line(answers={"urgent": True, "level": 2, "team": "data"})) == \
        {"urgent": 1, "level": 2, "team": 1}
    assert labels(questions, line(answers={"urgent": False})) == {"urgent": 0}


def test_unlabelled_lines_train_the_route_only(questions):
    train, _, test = training_rows(questions, [line()], [])
    row = (train + test)[0]
    assert (row["answers"], row["facts"], row["options"], row["option"]) == ({}, [], None, None)


@pytest.mark.parametrize("bad, problem", [
    ({"answers": {"mood": True}}, "unknown question 'mood'"),
    ({"answers": {"level": 9}}, "questions.level: 9 is not a level (0 to 2)"),
    ({"answers": {"team": "sales"}}, "questions.team: 'sales' is not one of reporting, data"),
    ({"answers": {"urgent": "yes"}}, "questions.urgent: needs true or false"),
    ({"route": "Tech", "options": REFUNDS, "option": "card"}, "routes.Tech has no options"),
    ({"options": REFUNDS, "option": "voucher"}, "option 'voucher' is not among the line's options"),
])
def test_bad_labels_name_the_line(questions, bad, problem):
    with pytest.raises(ValueError, match=re.escape(f"line 2: {problem}")):
        training_rows(questions, [line(), line(**bad)], [])


def test_ranked_lines_carry_their_options_and_facts(questions):
    train, _, test = training_rows(questions, [line(options=REFUNDS, option="credit",
                                                    facts=["The fare was refundable."])], [])
    row = (train + test)[0]
    assert (row["options"], row["option"], row["facts"]) == (REFUNDS, 1, ["The fare was refundable."])
