"""Tests for the CrowdStrike example's policy: the right answer for each booking."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from dataclasses import replace

import pytest
from test_crowdstrike_world import flight

from passengers import Booking, Fare
from policy import ROUTES, Label, answer
from world import World

ROUTE_NAMES = ["Rebook", "Rebook_Partner", "Overnight", "Refund", "Agent", "No_Action"]


def booking(**changes):
    base = Booking(id="B1", party=1, loyalty="none", legs=("X",), origin="A", dest="B",
                   fare=Fare(refundable=False, partner_ok=False), deadline=None, deadline_kind="none",
                   purpose="going home", still_travelling=True, vulnerable=False, travel_mode="air", upset="calm",
                   vague=False, known=0, sent=0)
    return replace(base, **changes)


@pytest.fixture
def world():
    return World([
        flight("X", "A", "B", 600, 700, cancelled=True),             # the booking's own flight, cancelled
        flight("T1", "A", "B", 800, 900),                             # our airline, today
        flight("P1", "A", "B", 750, 850, carrier="AS"),               # a partner, today
        flight("M1", "A", "B", 1500, 1600),                           # our airline, tomorrow morning
        # Other airlines' passengers, so these are no alternative for the Delta booking above
        flight("OK", "A", "B", 600, 700, carrier="UA", actual=(630, 730)),     # half an hour late
        flight("LATE", "A", "B", 600, 700, carrier="AA", actual=(750, 850)),   # two and a half hours late
        flight("SLOW", "A", "B", 610, 710, carrier="B6", actual=(640, 740)),
        flight("ON", "B", "C", 760, 860, carrier="B6"), flight("ON2", "B", "C", 800, 900, carrier="B6"),
    ], offsets={"A": 0, "B": 0, "C": 0})


def fill(world, *flights):
    for fid in flights:
        world.hold(f"full-{fid}", (fid,), world.free(fid))


def test_routes_descriptions_come_from_policy_md():
    assert list(ROUTES) == ROUTE_NAMES
    assert all(len(text) > 40 and "\n\n" not in text for text in ROUTES.values())


def test_rule_1_vague_goes_to_an_agent(world):
    assert answer(booking(vague=True, still_travelling=False), world, 0) == Label("Agent", None)


def test_rule_2_no_longer_travelling_is_a_refund(world):
    assert answer(booking(still_travelling=False), world, 0) == Label("Refund", None)


def test_rule_3_a_short_delay_needs_nothing(world):
    assert answer(booking(legs=("OK",)), world, 0) == Label("No_Action", None)


def test_rule_6_a_long_delay_with_nothing_faster_keeps_the_flight(world):
    assert answer(booking(legs=("LATE",)), world, 0) == Label("No_Action", None)


def test_rule_4_a_long_delay_rebooks_onto_a_faster_flight():
    faster = World([flight("LATE", "A", "B", 600, 700, actual=(750, 850)), flight("E1", "A", "B", 680, 780)],
                   offsets={"A": 0, "B": 0})
    assert answer(booking(legs=("LATE",)), faster, 0) == Label("Rebook", ("E1",))


def test_rule_4_our_airline_today(world):
    assert answer(booking(), world, 0) == Label("Rebook", ("T1",))
    assert answer(booking(deadline=950, deadline_kind="firm"), world, 0) == Label("Rebook", ("T1",))


def test_rule_5_a_partner_when_ours_misses_the_deadline(world):
    firm = booking(deadline=870, deadline_kind="firm", fare=Fare(refundable=False, partner_ok=True))
    assert answer(firm, world, 0) == Label("Rebook_Partner", ("P1",))
    # Without partner_ok the deadline can't be met: rule 8
    assert answer(replace(firm, fare=Fare(False, False)), world, 0) == Label("Agent", None)


def test_rule_7_overnight_when_nothing_today(world):
    fill(world, "T1")
    assert answer(booking(), world, 0) == Label("Overnight", ("M1",))
    assert answer(booking(deadline_kind="preferred", deadline=900), world, 0) == Label("Overnight", ("M1",))


def test_rule_8_a_firm_deadline_nothing_meets(world):
    firm = booking(deadline=800, deadline_kind="firm")
    assert answer(replace(firm, fare=Fare(True, False)), world, 0) == Label("Refund", None)
    assert answer(firm, world, 0) == Label("Agent", None)


def test_rule_9_otherwise_an_agent(world):
    fill(world, "T1", "M1")
    assert answer(booking(), world, 0) == Label("Agent", None)


def test_the_whole_party_needs_seats(world):
    world.hold("others", ("T1",), world.free("T1") - 1)
    assert answer(booking(party=2), world, 0) == Label("Overnight", ("M1",))


def test_a_missed_connection_rebooks_from_where_they_are(world):
    late = booking(legs=("SLOW", "ON"), dest="C")
    assert answer(late, world, 0) == Label("Rebook", ("SLOW", "ON2"))
    # Once SLOW has left they are on it: only the onward flight changes
    assert answer(late, world, 700) == Label("Rebook", ("ON2",))


def test_departed_alternatives_are_not_offered(world):
    assert answer(booking(), world, 801) == Label("Overnight", ("M1",))
