"""Tests for the CrowdStrike demo: the coordinator's clock, its live labels, and the scoreboard."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from dataclasses import replace

import pytest
from test_crowdstrike_policy import booking
from test_crowdstrike_world import flight

from policy import Label
from replay_day import LiveLabels, batches
from scoreboard import Scoreboard
from world import World


@pytest.fixture
def world():
    return World([flight("X", "A", "B", 600, 700, cancelled=True), flight("T1", "A", "B", 800, 900, capacity=20)],
                 offsets={"A": 0, "B": 0})


def held(legs, status="done"):
    return {"route": "Rebook", "option": {"id": "+".join(legs)},
            "actions": [{"tool": "hold_seats", "status": status, **({"result": {"held": list(legs)}}
                                                                     if status == "done" else {"error": "gone"})}]}


def test_batches_follow_the_clock_and_the_cap():
    made = [booking(id=name, sent=sent) for name, sent in
            [("e", 100), ("a", 10), ("c", 59), ("b", 20), ("d", 70)]]
    assert [[one.id for one in group] for group in batches(made, 2)] == [["a", "b"], ["c"], ["d", "e"]]


def test_a_batch_is_labelled_before_its_holds(world):
    world.hold("others", ("T1",), world.free("T1") - 1)        # one seat left on T1
    first, second = booking(id="P1"), booking(id="P2")
    live = LiveLabels(world)
    assert live.label_batch([first, second]) == [Label("Rebook", ("T1",)), Label("Rebook", ("T1",))]
    live.apply(first, held(("T1",)))
    live.apply(second, held(("T1",), status="failed"))
    assert (world.free("T1"), live.diverged) == (0, 0)


def test_a_hold_the_coordinator_cannot_make_is_counted_as_diverged(world):
    world.hold("others", ("T1",), world.free("T1"))
    live = LiveLabels(world)
    live.apply(booking(id="P1"), held(("T1",)))
    assert live.diverged == 1


def test_a_failed_hold_changes_nothing(world):
    live = LiveLabels(world)
    before = world.free("T1")
    live.apply(replace(booking(), party=3), held(("T1",), status="failed"))
    live.apply(booking(id="P9"), {"route": "Refund", "actions": []})
    assert (world.free("T1"), live.diverged) == (before, 0)


def reply(route, uncertain=False, option=None, actions=()):
    return {"route": route, "uncertain": uncertain, "option": None if option is None else {"id": option},
            "actions": [{"tool": tool, "status": status} for tool, status in actions]}


@pytest.fixture
def board():
    found = Scoreboard()
    made = [booking(id=f"P{i}", party=party) for i, party in enumerate([1, 2, 1, 3])]
    labels = [Label("Rebook", ("T1",)), Label("Agent", None), Label("Refund", None), Label("Rebook", ("T1",))]
    replies = [reply("Rebook", option="T1", actions=[("hold_seats", "done"), ("draft_reply", "done")]),
               reply("Agent"),
               reply("Rebook", uncertain=True, option="T1",
                     actions=[("hold_seats", "skipped"), ("draft_reply", "skipped")]),
               reply("Rebook", option="T1", actions=[("hold_seats", "failed"), ("draft_reply", "failed")])]
    found.add(made, labels, replies, seconds=2.0)
    return found


def test_scoreboard_counts_routes_options_and_handoffs(board):
    report = board.report()
    assert (report["bookings"], report["passengers"]) == (4, 7)
    assert report["route"] == {"Rebook": (2, 2), "Agent": (1, 1), "Refund": (0, 1)}
    assert report["option"] == {"Rebook": (2, 2)}
    assert report["automatic"] == 1
    assert report["handed"] == {"total": 3, "Agent route": 1, "uncertain": 1, "action failed": 1}
    assert report["actions"]["hold_seats"] == {"done": 1, "failed": 1, "skipped": 1}
    assert report["confidently_wrong"] == 0
    assert report["per_second"] == 2.0


def test_render_names_every_count(board):
    panel = board.render("10:15 UTC")
    for words in ("10:15 UTC", "4 bookings", "7 passengers", "2.0 decisions a second", "automatic 1",
                  "to a person 3", "uncertain 1", "Agent route 1", "action failed 1", "holds 1 done, 1 failed",
                  "Rebook 2 of 2", "Refund 0 of 1", "confidently wrong 0"):
        assert words in panel, words
