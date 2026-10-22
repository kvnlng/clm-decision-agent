"""The world of 19 July 2024: the real flights, made-up seats, and a search for alternatives.

Times are minutes from 19 July 2024 00:00 UTC. The flights come from flights.csv.gz (see README.md);
each flight's capacity is made up from a seed, and the seats left are shared by everyone deciding at
once, so two passengers can't take the same last seat.
"""
import csv
import gzip
import random
import threading
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

HERE = Path(__file__).parent
LOAD_FACTOR = 0.85
REGIONAL = {"9E", "OO", "YX", "OH", "MQ", "ZW", "G7", "C5", "QX", "PT"}
# This demo's airline groups: a regional carrier flying for a mainline airline counts as that airline.
# Carriers not listed are a group of their own
GROUPS = {"DL": "Delta", "9E": "Delta", "UA": "United", "AA": "American", "OH": "American", "MQ": "American"}
# This demo's partner airlines (made up: not any real agreement)
PARTNERS = {"Delta": {"AS"}, "United": {"B6"}, "American": {"AS"}}
CANCELLED_KNOWN = 120        # a cancellation is known this many minutes before scheduled departure
DELAY_KNOWN = 60             # and a long delay this many
LONG_DELAY = 120             # minutes late that count as a long delay
CONNECTION = 45              # least minutes between flights of the same airline
CONNECTION_CHANGE = 60       # and between different airlines


class SeatsGone(Exception):
    """A hold failed: the flight named has too few seats left."""

    def __init__(self, flight: str, free: int, party: int):
        super().__init__(f"{flight} has {free} seats left, not {party}")
        self.flight = flight


@dataclass(frozen=True, slots=True)
class Flight:  # pylint: disable=too-many-instance-attributes
    """One flight. `dep` and `arr` are when it really left and landed, None when cancelled (or, for
    `arr`, diverted)."""
    id: str
    carrier: str
    number: str
    tail: str
    origin: str
    dest: str
    sched_dep: int
    sched_arr: int
    dep: int | None
    arr: int | None
    cancelled: bool
    diverted: bool
    distance: int
    capacity: int

    @property
    def leaves(self) -> int:
        """When it leaves, as far as anyone can tell."""
        return self.sched_dep if self.dep is None else self.dep

    @property
    def lands(self) -> int:
        """When it lands, as far as anyone can tell."""
        return self.sched_arr if self.arr is None else self.arr


@dataclass(frozen=True)
class Journey:
    """One way to get there: one flight, or two with a connection."""
    legs: tuple[str, ...]
    dep: int
    arr: int

    def text(self, world: "World") -> str:
        """The journey as a sentence, in the airports' local times."""
        flights = [world.flights[leg] for leg in self.legs]
        parts = [f"{f.carrier}{f.number} from {f.origin} at {world.local(f.origin, f.leaves)}" if i == 0
                 else f"then {f.carrier}{f.number} at {world.local(f.origin, f.leaves)}"
                 for i, f in enumerate(flights)]
        stops = [f" to {f.origin}" for f in flights[1:]]
        lines = [part + (stops[i] if i < len(stops) else "") for i, part in enumerate(parts)]
        last = flights[-1]
        return f"{', '.join(lines)} to {last.dest}, arriving {world.local(last.dest, last.lands)} (local times)"


def group(carrier: str) -> str:
    """The demo's airline group a carrier belongs to."""
    return GROUPS.get(carrier, carrier)


def carriers_of(name: str) -> set[str]:
    """The carriers flying for a group."""
    return {carrier for carrier, owner in GROUPS.items() if owner == name} or {name}


def capacity(flight_id: str, carrier: str, distance: int, seed: int) -> int:
    """Seats on a flight: a regional jet has 50 to 76, a mainline aircraft 120 to 190, larger the
    further it flies."""
    rng = random.Random(f"{seed}:{flight_id}")
    if carrier in REGIONAL:
        return rng.randint(50, 76)
    size = 120 + 70 * min(distance, 2500) / 2500 + rng.uniform(-15, 15)
    return max(120, min(190, round(size)))


@cache
def _rows() -> tuple[tuple[dict, ...], dict[str, int], dict[str, str]]:
    with gzip.open(HERE / "flights.csv.gz", "rt", encoding="utf-8", newline="") as file:
        rows = tuple(csv.DictReader(file))
    with open(HERE / "airports.csv", encoding="utf-8", newline="") as file:
        airports = list(csv.DictReader(file))
    return rows, {row["airport"]: int(row["utc_offset"]) for row in airports}, \
        {row["airport"]: row["city"] for row in airports}


class World:  # pylint: disable=too-many-instance-attributes
    """The day's flights and the seats left on them, shared by every decision."""

    def __init__(self, flights: list[Flight], offsets: dict[str, int], free: dict[str, int] | None = None,
                 cities: dict[str, str] | None = None):
        self.flights = {flight.id: flight for flight in flights}
        self.offsets = offsets
        self.cities = cities or {}
        self._free = {flight.id: flight.capacity - round(LOAD_FACTOR * flight.capacity) for flight in flights}
        self._free.update(free or {})
        self._lock = threading.Lock()
        self.holds: dict[str, tuple[tuple[str, ...], int]] = {}
        # Each route's flights in the order they leave, for the search
        routes: dict[tuple[str, str], list[Flight]] = {}
        for flight in flights:
            if not flight.cancelled and not flight.diverted:
                routes.setdefault((flight.origin, flight.dest), []).append(flight)
        self._routes = {pair: sorted(found, key=lambda f: f.leaves) for pair, found in routes.items()}
        self._leaving = {pair: [f.leaves for f in found] for pair, found in self._routes.items()}
        self._onward: dict[str, set[str]] = {}
        for origin, dest in self._routes:
            self._onward.setdefault(origin, set()).add(dest)

    @classmethod
    def load(cls, seed: int) -> "World":
        """The real day, with capacities made up from `seed`."""
        rows, offsets, cities = _rows()
        flights = [Flight(id=row["id"], carrier=row["carrier"], number=row["number"], tail=row["tail"],
                          origin=row["origin"], dest=row["dest"], sched_dep=int(row["sched_dep"]),
                          sched_arr=int(row["sched_arr"]), dep=int(row["dep"]) if row["dep"] else None,
                          arr=int(row["arr"]) if row["arr"] else None, cancelled=row["cancelled"] == "1",
                          diverted=row["diverted"] == "1", distance=int(row["distance"]),
                          capacity=capacity(row["id"], row["carrier"], int(row["distance"]), seed))
                   for row in rows]
        return cls(flights, offsets, cities=cities)

    def local(self, airport: str, minute: int) -> str:
        """A moment as the airport's clock shows it, with the date when it isn't 19 July."""
        local = minute + self.offsets.get(airport, 0)
        day, clock = divmod(local, 1440)
        text = f"{clock // 60:02d}:{clock % 60:02d}"
        return text if day == 0 else f"{text} on {19 + day} July"

    def known_at(self, flight: Flight) -> int | None:
        """The latest moment a flight's disruption is known, or None when it isn't disrupted."""
        if flight.cancelled:
            return flight.sched_dep - CANCELLED_KNOWN
        if flight.dep is not None and flight.dep - flight.sched_dep >= LONG_DELAY:
            return flight.sched_dep - DELAY_KNOWN
        return None

    def free(self, flight_id: str) -> int:
        """Seats left on a flight."""
        return self._free[flight_id]

    def hold(self, booking_id: str, legs: tuple[str, ...], party: int) -> None:
        """Hold `party` seats on every leg, or on none: SeatsGone names the first leg short of seats."""
        if party < 1:
            raise ValueError(f"a party is at least 1 person, not {party}")
        for leg in legs:
            if leg not in self._free:
                raise ValueError(f"unknown flight {leg!r}")
        with self._lock:
            if booking_id in self.holds:
                raise ValueError(f"booking {booking_id!r} already holds seats; release them first")
            for leg in legs:
                if self._free[leg] < party:
                    raise SeatsGone(leg, self._free[leg], party)
            for leg in legs:
                self._free[leg] -= party
            self.holds[booking_id] = (tuple(legs), party)

    def release(self, booking_id: str) -> None:
        """Give back the seats a booking holds."""
        with self._lock:
            legs, party = self.holds.pop(booking_id)
            for leg in legs:
                self._free[leg] += party

    def _departures(self, origin: str, dest: str, after: int, party: int, carriers: set[str]):
        found = self._routes.get((origin, dest), [])
        for flight in found[bisect_left(self._leaving.get((origin, dest), []), after):]:
            if flight.carrier in carriers and self._free[flight.id] >= party:
                yield flight

    def _connecting(self, origin: str, dest: str, start: int, party: int, carriers: set[str]) -> list[Journey]:
        found = []
        for stop in self._onward.get(origin, ()):
            if stop == dest or (stop, dest) not in self._routes:
                continue
            for first in self._departures(origin, stop, start, party, carriers):
                for second in self._departures(stop, dest, first.lands + CONNECTION, party, carriers):
                    gap = CONNECTION if group(first.carrier) == group(second.carrier) else CONNECTION_CHANGE
                    if second.leaves - first.lands >= gap:
                        found.append(Journey((first.id, second.id), first.leaves, second.lands))
                        # The first that connects: a later one from here almost never lands sooner
                        break
        return found

    def search(self, origin: str, dest: str, after: int, now: int, party: int,  # pylint: disable=too-many-arguments,too-many-positional-arguments
               carriers: set[str], *, deadline: int | None = None, limit: int = 3,
               accept: Callable[[Journey], bool] | None = None) -> list[Journey]:
        """Up to `limit` journeys leaving at or after both `after` and `now`, earliest arrival first, on
        the given carriers, with seats for the whole party, no more than one connection, and arriving by
        `deadline` if given, and passing `accept` if given. No two journeys share a flight, so each is a
        different choice."""
        start = max(after, now)
        found = [Journey((f.id,), f.leaves, f.lands) for f in self._departures(origin, dest, start, party, carriers)]
        found += self._connecting(origin, dest, start, party, carriers)
        found = [j for j in found if (deadline is None or j.arr <= deadline) and (accept is None or accept(j))]
        found.sort(key=lambda j: (j.arr, len(j.legs), j.dep, j.legs))
        chosen: list[Journey] = []
        used: set[str] = set()
        for journey in found:
            if not used.intersection(journey.legs):
                chosen.append(journey)
                used.update(journey.legs)
                if len(chosen) == limit:
                    break
        return chosen
