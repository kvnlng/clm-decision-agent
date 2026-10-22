"""Tests for the CrowdStrike comparison: language models given the same inputs as CLM."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
import json

import pytest
from test_crowdstrike_policy import booking
from test_crowdstrike_tools import SCENARIO, use
from test_crowdstrike_tools import world as _world  # pylint: disable=unused-import

import compare_models
from compare_models import inputs, parse, prompt, summary
from passengers import Fare

world = _world


@pytest.fixture
def case(world):
    one = booking(fare=Fare(refundable=False, partner_ok=True))
    use(world, one)
    return {"text": "My flight was cancelled, please rebook me.", "context": {"booking": one.id, "now": 0},
            "route": "Refund", "answers": {"upset": 1}}


def test_inputs_offer_every_rebooking_routes_options(case):
    found = inputs(case)
    assert {route: [option["id"] for option in options] for route, options in found["options"].items()} == \
        {"Rebook": ["T1"], "Rebook_Partner": ["P1"], "Overnight": ["M1"]}
    assert found["message"] == case["text"] and found["facts"][1] == "DLX is cancelled."


def test_prompt_system_part_is_the_same_for_every_case(case):
    first, second = inputs(case), {**inputs(case), "message": "Something else entirely."}
    assert prompt(first)[0] == prompt(second)[0]
    assert "Rebook_Partner" in prompt(first)[0] and "T1" in prompt(first)[1] and "DLX is cancelled." in prompt(first)[1]


def test_parse_reads_route_and_option(case):
    found = inputs(case)
    assert parse(json.dumps({"route": "Rebook", "option": "T1"}), found) == \
        ({"route": "Rebook", "uncertain": False, "option": {"id": "T1"}, "answers": {}}, True)
    assert parse(json.dumps({"route": "Refund", "option": ""}), found)[0]["option"] is None


@pytest.mark.parametrize("text", ["not json", json.dumps({"route": "Upgrade", "option": ""}),
                                  json.dumps({"route": "Rebook", "option": "P1"}), "", json.dumps([1])])
def test_parse_marks_unusable_replies(case, text):
    reply, usable = parse(text, inputs(case))
    assert not usable and reply["route"] == "unusable"


def test_summary_scales_cost_and_time_to_the_day(monkeypatch):
    monkeypatch.setitem(compare_models.PRICES, "test-model", {"input": 1.0, "output": 5.0, "cache_write": 1.25,
                                                               "cache_read": 0.1})
    cases = [{"route": "Rebook", "option": "T1", "answers": {}}, {"route": "Refund", "answers": {}}]
    replies = [{"route": "Rebook", "uncertain": False, "option": {"id": "T1"}, "answers": {}},
               {"route": "unusable", "uncertain": False, "option": None, "answers": {}}]
    usage = [{"input": 1000, "output": 100, "cache_write": 0, "cache_read": 2000}] * 2
    found = summary("test-model", cases, replies, usage, seconds=4.0, day=1000)
    assert found["route"] == {"Rebook": (1, 1), "Refund": (0, 1)}
    assert found["unusable"] == 1
    assert found["seconds_per_decision"] == 2.0 and found["day_hours"] == pytest.approx(2000 / 3600)
    # 1000 × $1 + 100 × $5 + 2000 × $0.10, per million tokens
    assert found["cost_per_decision"] == pytest.approx(0.0017)
    assert found["day_cost"] == pytest.approx(1.7)
    assert found["day_weighted"] == pytest.approx(compare_models.SHARES["Rebook"])


def test_missing_credentials_skip_claude(monkeypatch, capsys, tmp_path):
    def no_client():
        raise RuntimeError("no credentials")

    monkeypatch.setattr(compare_models, "claude_client", no_client)
    assert compare_models.main([str(SCENARIO), "--models", "haiku", "--limit", "1", "--runs", str(tmp_path)]) == 0
    assert "Skipping haiku: no Anthropic credentials (RuntimeError: no credentials)" in capsys.readouterr().out


def test_credentials_are_checked_before_any_case(monkeypatch, capsys, tmp_path):
    class Models:
        def retrieve(self, model_id):
            raise RuntimeError(f"401 for {model_id}")

    class Client:
        models = Models()

    monkeypatch.setattr(compare_models, "claude_client", Client)
    assert compare_models.main([str(SCENARIO), "--models", "sonnet", "--limit", "1", "--runs", str(tmp_path)]) == 0
    assert "Skipping sonnet: no Anthropic credentials (RuntimeError: 401 for claude-sonnet-5-5)" in \
        capsys.readouterr().out
