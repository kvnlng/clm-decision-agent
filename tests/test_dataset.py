"""Tests for dataset.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,unused-argument
from dataset import examples_path, load_examples, save_examples, split_of, training_rows


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
    records = [{"kind": "route", "state_text": "User: my app crashes", "human": "Tech",
                "clm": {"route": "Billing", "confidence": 0.8, "probabilities": {}}},
               {"kind": "reply", "state_text": "x", "route": "Tech", "reply": "", "human": {"acceptable": True}}]
    train, reviewed, test = training_rows(scenario, examples, records)

    assert reviewed == [("User: my app crashes", "Tech")]
    train = train + reviewed
    assert all(route in scenario.routes for _, route in train + test)
    assert {state for state, _ in train + test} == {f"User: {t}" for t in texts} | {"User: my app crashes"}
    assert {state for state, _ in test} == {f"User: {t}" for t in texts if split_of(t) == "test"}
    assert not {state for state, _ in train} & {state for state, _ in test}


def test_a_reviewed_route_that_repeats_a_held_back_example_is_not_trained_on(scenario):
    texts = [f"message number {i}" for i in range(40)]
    held = next(t for t in texts if split_of(t) == "test")
    records = [{"kind": "route", "state_text": f"User: {held}", "human": "Billing",
                "clm": {"route": "Tech", "confidence": 0.8, "probabilities": {}}}]
    _, reviewed, _ = training_rows(scenario, [{"text": t, "route": "Billing"} for t in texts], records)
    assert reviewed == []
