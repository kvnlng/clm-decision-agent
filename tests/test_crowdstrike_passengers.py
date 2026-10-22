"""Tests for the CrowdStrike example's passengers."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import time
from collections import Counter

import pytest
from test_crowdstrike_world import flight

from passengers import CORRIDOR, STARTING, make_bookings
from world import LOAD_FACTOR, World, group

DAY = "-2024-07-19-"


def small_world():
    return World([
        flight(f"X{DAY}A", "A", "B", 600, 700, cancelled=True),
        flight(f"OK{DAY}A", "A", "B", 600, 700),
        flight(f"LATE{DAY}A", "A", "B", 600, 700, actual=(750, 850)),
        flight(f"SLOW{DAY}A", "A", "B", 610, 710, actual=(640, 740)),
        # Onward flights, large enough to have room for passengers connecting from all four
        flight(f"ON{DAY}B", "B", "C", 760, 860, capacity=400),     # 20 minutes after SLOW lands late: missed
        flight(f"ON2{DAY}B", "B", "C", 800, 900, capacity=400),
    ], offsets={"A": 0, "B": 0, "C": 0})


def sold(world, fid):
    return round(LOAD_FACTOR * world.flights[fid].capacity)


def test_bookings_are_deterministic_per_seed():
    assert make_bookings(small_world(), 0, 1.0) == make_bookings(small_world(), 0, 1.0)
    assert make_bookings(small_world(), 0, 1.0) != make_bookings(small_world(), 1, 1.0)


def test_only_affected_bookings_are_made():
    world = small_world()
    bookings = make_bookings(world, 0, 1.0)
    reasons = set()
    for booking in bookings:
        first = world.flights[booking.legs[0]]
        legs = [world.flights[leg] for leg in booking.legs]
        missed = len(legs) == 2 and not first.cancelled and legs[1].leaves - first.lands < 45
        assert any(world.known_at(leg) is not None for leg in legs) or missed
        reasons.add("missed" if missed else first.id.split("-")[0])
    assert {"X", "LATE", "missed"} <= reasons
    starting = sum(b.party for b in bookings if b.legs[0] == f"X{DAY}A")
    assert starting == round(STARTING * sold(world, f"X{DAY}A"))


def test_connections_are_real_and_legal():
    world = small_world()
    connecting = [b for b in make_bookings(world, 0, 1.0) if len(b.legs) == 2]
    assert connecting
    for booking in connecting:
        first, onward = (world.flights[leg] for leg in booking.legs)
        assert onward.origin == first.dest and group(onward.carrier) == group(first.carrier)
        assert 45 <= onward.sched_dep - first.sched_arr <= 180
        assert (booking.origin, booking.dest) == (first.origin, onward.dest)


def test_bookings_fit_in_the_sold_seats():
    world = small_world()
    seats = Counter()
    for booking in make_bookings(world, 0, 1.0):
        for leg in booking.legs:
            seats[leg] += booking.party
    assert all(seats[fid] <= sold(world, fid) for fid in seats)
    assert all(world.free(fid) == f.capacity - sold(world, fid) for fid, f in world.flights.items())


def test_only_the_contact_share_write():
    world = small_world()
    everyone, some = make_bookings(world, 0, 1.0), make_bookings(world, 0, 0.25)
    assert 0.15 < len(some) / len(everyone) < 0.35


@pytest.fixture(scope="module")
def day():
    world = World.load(0)
    start = time.perf_counter()
    bookings = make_bookings(world, 0)
    print(f"\n{len(bookings)} bookings write in, {sum(b.party for b in bookings)} passengers, "
          f"made in {time.perf_counter() - start:.1f} s")
    return world, bookings


def share(bookings, test):
    return sum(map(test, bookings)) / len(bookings)


def test_attribute_shares_are_near_their_targets(day):
    _, bookings = day
    targets = [(lambda b: b.deadline_kind == "none", 0.5), (lambda b: b.deadline_kind == "firm", 0.2),
               (lambda b: b.still_travelling, 0.85), (lambda b: b.vulnerable, 0.1),
               (lambda b: b.upset == "angry", 0.2), (lambda b: b.upset == "calm", 0.4), (lambda b: b.vague, 0.08),
               (lambda b: b.loyalty == "gold", 0.1), (lambda b: b.fare.refundable, 0.3),
               (lambda b: b.fare.partner_ok, 0.6), (lambda b: b.travel_mode == "either", 0.2)]
    for test, target in targets:
        assert abs(share(bookings, test) - target) < 0.03
    assert all(b.origin in CORRIDOR and b.dest in CORRIDOR for b in bookings if b.travel_mode == "rail")
    assert all((b.deadline is None) == (b.deadline_kind == "none") for b in bookings)


def test_arrivals_come_in_bursts(day):
    _, bookings = day
    assert [b.sent for b in bookings] == sorted(b.sent for b in bookings)
    assert all(b.sent >= b.known * 60 for b in bookings)
    minutes = Counter(b.sent // 60 for b in bookings)
    assert max(minutes.values()) >= 5 * len(bookings) / len(minutes)
