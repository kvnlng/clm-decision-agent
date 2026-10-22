"""Tests for the CrowdStrike example's world: the flight extract, capacities, search and seats."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import csv
import gzip
import threading
import time
from pathlib import Path

import pytest

from world import Flight, World

EXTRACT = Path(__file__).parent.parent / "examples" / "crowdstrike" / "flights.csv.gz"


def extract_rows():
    with gzip.open(EXTRACT, "rt", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_extract_times_are_utc_and_consistent():
    rows = extract_rows()
    assert all(int(r["sched_arr"]) - int(r["sched_dep"]) == int(r["elapsed"]) for r in rows)
    day = [r for r in rows if "-2024-07-19-" in r["id"]]
    # BTS has 21,468 that day; 8 are dropped for schedules that contradict themselves (seven list the same
    # arrival as departure time). None of them was cancelled
    assert (len(day), sum(r["cancelled"] == "1" for r in day)) == (21460, 3548)
    flight = next(r for r in rows if r["id"] == "DL827-2024-07-19-ATL")
    # 06:15 in Atlanta (UTC-4) and 07:51 in Los Angeles (UTC-7); it left at 10:19 Atlanta time
    assert (int(flight["sched_dep"]), int(flight["sched_arr"]), int(flight["dep"])) == (615, 891, 859)
    assert flight["dest"] == "LAX" and flight["tail"] == "N121DZ"
    tomorrow = [r for r in rows if "-2024-07-20-" in r["id"]]
    assert tomorrow and all(int(r["sched_dep"]) < 1440 + 16 * 60 + 12 * 60 for r in tomorrow)


def flight(fid, origin, dest, dep, arr, carrier="DL", cancelled=False, actual=None, capacity=150):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """A hand-made flight; `actual` is (dep, arr) when it ran late."""
    real = actual or (None if cancelled else (dep, arr))
    return Flight(id=fid, carrier=carrier, number=fid, tail="N1", origin=origin, dest=dest, sched_dep=dep,
                  sched_arr=arr, dep=None if real is None else real[0], arr=None if real is None else real[1],
                  cancelled=cancelled, diverted=False, distance=500, capacity=capacity)


@pytest.fixture
def small():
    flights = [flight("AC", "A", "C", 100, 400), flight("AB", "A", "B", 100, 200),
               flight("BC1", "B", "C", 240, 300),            # 40 minutes after AB lands: too tight
               flight("BC2", "B", "C", 250, 320), flight("AC_X", "A", "C", 90, 250, cancelled=True),
               flight("AC_UA", "A", "C", 120, 260, carrier="UA")]
    return World(flights, offsets={"A": 0, "B": 0, "C": 0})


def test_capacity_is_seeded_and_regional_jets_are_small():
    day = World.load(0)
    regional = [f for f in day.flights.values() if f.carrier == "9E"]
    mainline = [f for f in day.flights.values() if f.carrier == "DL"]
    assert regional and all(50 <= f.capacity <= 76 for f in regional)
    assert mainline and all(120 <= f.capacity <= 190 for f in mainline)
    assert [f.capacity for f in day.flights.values()] == [f.capacity for f in World.load(0).flights.values()]
    assert [f.capacity for f in day.flights.values()] != [f.capacity for f in World.load(1).flights.values()]
    some = mainline[0]
    assert day.free(some.id) == some.capacity - round(0.85 * some.capacity)


def test_search_finds_the_earliest_arrival_and_respects_connections(small):
    found = small.search("A", "C", after=0, now=0, party=1, carriers={"DL"})
    assert [j.legs for j in found] == [("AB", "BC2"), ("AC",)]
    assert (found[0].dep, found[0].arr) == (100, 320)
    assert found[0].text(small) == ("DLAB from A at 01:40 to B, then DLBC2 at 04:10 to C, "
                                    "arriving 05:20 (local times)")
    assert small.search("A", "C", after=0, now=0, party=1, carriers={"DL"}, deadline=310) == []


def test_search_skips_departed_cancelled_and_full_flights(small):
    assert small.search("A", "C", after=0, now=101, party=1, carriers={"DL"}) == []
    assert [j.legs for j in small.search("A", "C", after=0, now=0, party=1, carriers={"UA"})] == [("AC_UA",)]
    small.hold("someone", ("BC2",), small.free("BC2"))
    assert [j.legs for j in small.search("A", "C", after=0, now=0, party=1, carriers={"DL"})] == [("AC",)]
    assert small.search("A", "C", after=0, now=0, party=small.free("AC") + 1, carriers={"DL"}) == []


def test_two_threads_race_for_the_last_seats():
    world = World([flight("F", "A", "B", 100, 200)], offsets={"A": 0, "B": 0}, free={"F": 10})
    results = []
    threads = [threading.Thread(target=lambda i=i: results.append(world.hold(f"b{i}", ("F",), 2)))
               for i in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert (results.count(True), world.free("F")) == (5, 0)


def test_a_failed_hold_changes_nothing(small):
    small.hold("first", ("BC2",), small.free("BC2"))
    before = small.free("AB")
    assert small.hold("second", ("AB", "BC2"), 1) is False
    assert small.free("AB") == before


def test_known_at_follows_the_clock():
    world = World([flight("X", "A", "B", 600, 700, cancelled=True),
                   flight("L", "A", "B", 600, 700, actual=(750, 850)),
                   flight("S", "A", "B", 600, 700, actual=(630, 730))], offsets={"A": 0, "B": 0})
    assert [world.known_at(world.flights[f]) for f in ("X", "L", "S")] == [480, 540, None]


def test_journeys_say_their_times_in_local_time():
    world = World([flight("F", "ATL", "LAX", 615, 891)], offsets={"ATL": -240, "LAX": -420})
    journeys = world.search("ATL", "LAX", after=0, now=0, party=1, carriers={"DL"})
    assert [j.text(world) for j in journeys] == ["DLF from ATL at 06:15 to LAX, arriving 07:51 (local times)"]


def test_a_search_over_the_whole_day_is_fast():
    day = World.load(0)
    start = time.perf_counter()
    for _ in range(20):
        day.search("ATL", "SEA", after=600, now=600, party=2, carriers={"DL", "9E"})
    assert (time.perf_counter() - start) / 20 < 0.02


def test_airports_with_no_direct_flight_are_reached_by_a_connection(small):
    found = small.search("A", "B", after=0, now=0, party=1, carriers={"DL"})
    assert [j.legs for j in found] == [("AB",)]
    world = World([flight("AB", "A", "B", 100, 200), flight("BD", "B", "D", 260, 330)], offsets={})
    assert [j.legs for j in world.search("A", "D", after=0, now=0, party=1, carriers={"DL"})] == [("AB", "BD")]


def test_airports_have_city_names():
    day = World.load(0)
    assert (day.cities["ATL"], day.cities["LAX"], day.cities["JFK"]) == ("Atlanta", "Los Angeles", "New York")
