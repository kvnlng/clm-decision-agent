"""Tests for synthesize.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,unused-argument
from pathlib import Path

import pytest
from conftest import BASE_TOML, FakeGenerator

from dataset import load_examples
from scenario import load_scenario
from synthesize import STYLES, build_prompt, parse_messages, synthesize


def test_parse_messages_accepts_a_json_list_with_text_around_it():
    found = parse_messages('Sure:\n["I need a refund", " Charged twice! ", "", 3]\nDone.')
    assert found == ["I need a refund", "Charged twice!"]


def test_parse_messages_falls_back_to_lines():
    assert parse_messages("1. I need a refund\n- Charged twice\n\n") == ["I need a refund", "Charged twice"]


def test_prompt_names_the_route_and_what_it_is_not(scenario):
    prompt = build_prompt(scenario, scenario.routes["Billing"], STYLES[0], 8)
    assert "Charges and refunds" in prompt and "Bugs and outages" in prompt
    assert "8" in prompt and STYLES[0] in prompt


def test_prompt_asks_for_a_neutral_kind_of_message_unless_the_scenario_says(scenario, write_scenario):
    assert build_prompt(scenario, scenario.routes["Billing"], STYLES[0], 8).startswith(
        "Write 8 different messages that a person might send to an assistant.\n")

    told = load_scenario(write_scenario(BASE_TOML + '\n[synthesize]\nmessages = "emails to a billing team"\n'))
    assert build_prompt(told, told.routes["Billing"], STYLES[0], 8).startswith(
        "Write 8 different emails to a billing team.\n")


def test_the_example_keeps_its_help_desk_wording():
    example = load_scenario(Path(__file__).parent.parent / "examples" / "service_desk" / "scenario.toml")
    prompt = build_prompt(example, example.routes["Data_Access"], STYLES[0], 5)
    assert prompt.startswith("Write 5 different messages that staff might send to a help desk.\n")


def test_synthesize_writes_labelled_examples_without_repeats(scenario, tmp_path):
    gen = FakeGenerator(*['["refund please", "charged twice"]', '["refund please", "wrong amount"]'] * 2)
    path = tmp_path / "out.jsonl"
    count = synthesize(scenario, gen, path, per_route=3, per_call=2, write=lambda line: None)
    rows = load_examples(path)
    assert count == len(rows) == 6
    assert [r["route"] for r in rows] == ["Billing"] * 3 + ["Tech"] * 3
    assert [r["text"] for r in rows][:3] == ["refund please", "charged twice", "wrong amount"]


def test_synthesize_stops_when_the_generator_stops_producing_new_messages(scenario, tmp_path):
    gen = FakeGenerator(*['["same"]'] * 40)
    count = synthesize(scenario, gen, tmp_path / "out.jsonl", per_route=5, per_call=2,
                       write=lambda line: None)
    assert count == 2


def test_parse_messages_handles_one_list_per_line_and_nested_lists():
    assert parse_messages('["When will ticket 903 be resolved?"]\n["Who has ticket 112?"]') == \
        ["When will ticket 903 be resolved?", "Who has ticket 112?"]
    assert parse_messages('[["a b"], "c d"]') == ["a b", "c d"]


def test_parse_messages_drops_fragments_of_cut_off_output():
    assert parse_messages('[\n"I need a refund for invoice 12",\n"Charged tw') == ["I need a refund for invoice 12"]


BOTH = ["I need a refund today", "Charged twice again"]


@pytest.mark.parametrize("text, expected", [
    ("Here are 2 messages:\n1. I need a refund today\n2. Charged twice again", BOTH),
    ('```json\n[\n"I need a refund today",\n"Charged tw', ["I need a refund today"]),
    ('["I need a refund today", "Charged twice again",]', BOTH),
    ("[\u201cI need a refund today\u201d, \u201cCharged twice again\u201d]", BOTH),
    ("Here are [2] messages:\n1. I need a refund today\n2. Charged twice again", BOTH),
    ('[{"message": "I need a refund today"}, {"message": "Charged twice again"}]', BOTH),
    ('<think>They want [two] messages.</think>\n["I need a refund today"]', ["I need a refund today"]),
])
def test_parse_messages_copes_with_untidy_output(text, expected):
    assert parse_messages(text) == expected
