"""Tests for the CrowdStrike demo: the coordinator's clock, its live labels, and the scoreboard."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from dataclasses import replace

import pytest
from test_crowdstrike_policy import booking
from test_crowdstrike_world import flight

from policy import Label
from replay_day import LiveLabels, batches
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
