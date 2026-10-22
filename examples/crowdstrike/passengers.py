"""The passengers whose plans the day broke, and when each writes in.

Everything here is made up from a seed: who sits on each flight, who connects onward, and what each
party needs. The shares are this demo's choice:

- party size: 1 to 4, weighted 60/25/10/5;
- 30% of parties connect onward to a flight of the same airline 45 to 180 minutes after they land;
- deadline: none 50%, preferred 30%, firm 20%;
- still want to travel 85%; vulnerable travellers in the party 10%;
- way to travel: air 70%, either 20%, rail 10% (rail only between airports of the north-east corridor,
  otherwise air);
- upset: calm 40%, annoyed 40%, angry 20%; vague message 8%;
- loyalty: none 70%, silver 20%, gold 10%; refundable fare 30%; fare allows a partner airline 60%.

Airlines announce disruptions in waves, every 20 to 60 minutes from the outage at 04:10 UTC, so
messages arrive in bursts: most passengers write within minutes of hearing, some within an hour or
two, and a few hours later.
"""
import random
import string
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from world import CONNECTION, LOAD_FACTOR, Flight, World, group

DAY = "-2024-07-19-"
STARTING = 0.7               # a flight's sold seats taken by parties starting on it; the rest connect onto it
OUTAGE = 4 * 60 + 10         # 04:10 UTC
AHEAD = 6 * 60               # a cancellation is announced up to this long before its clock-rule time
PARTY_SIZES = ([1, 2, 3, 4], [60, 25, 10, 5])
CORRIDOR = {"BOS", "PVD", "BDL", "JFK", "LGA", "EWR", "HPN", "ISP", "PHL", "BWI", "DCA", "IAD", "RIC"}
PURPOSES = {
    "none": ["going home", "a holiday", "visiting family", "a trip with friends", "a weekend away"],
    "preferred": ["dinner with family", "a work meeting", "a concert", "a friend's birthday", "a hotel check-in"],
    "firm": ["a wedding", "a job interview", "a funeral", "a cruise departure", "surgery", "an exam",
             "my daughter's graduation", "a connecting international flight"],
}


@dataclass(frozen=True)
class Fare:
    """What the ticket allows."""
    refundable: bool
    partner_ok: bool


@dataclass(frozen=True)
class Booking:  # pylint: disable=too-many-instance-attributes
    """One party's booking, and how the day went for them."""
    id: str
    party: int
    loyalty: str
    legs: tuple[str, ...]
    origin: str
    dest: str
    fare: Fare
    deadline: int | None
    deadline_kind: str
    purpose: str
    still_travelling: bool
    vulnerable: bool
    travel_mode: str
    upset: str
    vague: bool
    known: int
    sent: int


def _pick(rng: random.Random, shares: list[tuple[str, float]]) -> str:
    roll, total = rng.random(), 0.0
    for value, part in shares:
        total += part
        if roll < total:
            return value
    return shares[-1][0]


class _Waves:
    """When each airline announces what."""

    def __init__(self, seed: int):
        self.seed = seed
        self.times: dict[str, list[int]] = {}

    def of(self, name: str) -> list[int]:
        """The airline's wave times."""
        if name not in self.times:
            rng, at, found = random.Random(f"{self.seed}:waves:{name}"), OUTAGE, []
            while at < 2 * 1440:
                found.append(at)
                at += rng.randint(20, 60)
            self.times[name] = found
        return self.times[name]

    def cancelled(self, name: str, latest: int) -> int:
        """A cancellation goes out in the first wave within AHEAD of its clock-rule time."""
        waves = self.of(name)
        return waves[min(bisect_left(waves, latest - AHEAD), len(waves) - 1)]

    def delayed(self, name: str, latest: int) -> int:
        """A delay goes out in the last wave at or before its clock-rule time."""
        waves = self.of(name)
        return waves[max(bisect_right(waves, latest) - 1, 0)]


class _Maker:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    def __init__(self, world: World, seed: int, contact_share: float):
        self.world, self.seed, self.contact_share = world, seed, contact_share
        self.waves = _Waves(seed)
        self.room = {fid: self._sold(f) - round(STARTING * self._sold(f)) for fid, f in world.flights.items()}
        self.leaving: dict[tuple[str, str], list[Flight]] = {}
        for found in sorted(world.flights.values(), key=lambda f: (f.sched_dep, f.id)):
            self.leaving.setdefault((found.origin, group(found.carrier)), []).append(found)
        self.times = {key: [f.sched_dep for f in found] for key, found in self.leaving.items()}
        self.ids: set[str] = set()

    @staticmethod
    def _sold(flight: Flight) -> int:
        return round(LOAD_FACTOR * flight.capacity)

    def _onward(self, first: Flight) -> list[Flight]:
        key = (first.dest, group(first.carrier))
        found = self.leaving.get(key, [])
        start = bisect_left(self.times.get(key, []), first.sched_arr + CONNECTION)
        end = bisect_right(self.times.get(key, []), first.sched_arr + 180)
        return [f for f in found[start:end] if f.dest != first.origin]

    def _known(self, legs: list[Flight]) -> int | None:
        """When the party hears their journey is broken, or None when it isn't."""
        first, heard = legs[0], []
        for leg in legs:
            latest = self.world.known_at(leg)
            if latest is not None:
                waves = self.waves.cancelled if leg.cancelled else self.waves.delayed
                heard.append(waves(group(leg.carrier), latest))
        if len(legs) == 2 and not first.cancelled and legs[1].leaves - first.lands < CONNECTION:
            heard.append(self.waves.delayed(group(first.carrier), first.sched_dep))
        return min(heard) if heard else None

    def _id(self, rng: random.Random) -> str:
        while True:
            code = "".join(rng.choices(string.ascii_uppercase + "23456789", k=6))
            if code not in self.ids:
                self.ids.add(code)
                return code

    def _booking(self, rng: random.Random, legs: list[Flight], party: int, known: int) -> Booking:
        first, last = legs[0], legs[-1]
        kind = _pick(rng, [("none", 0.5), ("preferred", 0.3), ("firm", 0.2)])
        deadline = None
        if kind != "none":
            local = last.sched_arr + rng.choice([60, 90, 120, 180, 240, 360]) + self.world.offsets.get(last.dest, 0)
            deadline = -(-local // 30) * 30 - self.world.offsets.get(last.dest, 0)
        mode = _pick(rng, [("air", 0.7), ("either", 0.2), ("rail", 0.1)])
        if mode == "rail" and not (first.origin in CORRIDOR and last.dest in CORRIDOR):
            mode = "air"
        delay = _pick(rng, [("soon", 0.6), ("hour", 0.3), ("later", 0.1)])
        minutes = {"soon": lambda: rng.expovariate(1 / 4), "hour": lambda: rng.uniform(15, 120),
                   "later": lambda: rng.uniform(120, 480)}[delay]()
        sent = known * 60 + round(minutes * 60)
        if not first.cancelled and (first.leaves - 10) * 60 >= known * 60:
            sent = min(sent, (first.leaves - 10) * 60)
        return Booking(
            id=self._id(rng), party=party, loyalty=_pick(rng, [("none", 0.7), ("silver", 0.2), ("gold", 0.1)]),
            legs=tuple(leg.id for leg in legs), origin=first.origin, dest=last.dest,
            fare=Fare(refundable=rng.random() < 0.3, partner_ok=rng.random() < 0.6),
            deadline=deadline, deadline_kind=kind, purpose=rng.choice(PURPOSES[kind]),
            still_travelling=rng.random() < 0.85, vulnerable=rng.random() < 0.1, travel_mode=mode,
            upset=_pick(rng, [("calm", 0.4), ("annoyed", 0.4), ("angry", 0.2)]), vague=rng.random() < 0.08,
            known=known, sent=sent)

    def from_flight(self, first: Flight) -> list[Booking]:
        """The bookings starting on a flight that the day broke, and that write in."""
        rng = random.Random(f"{self.seed}:{first.id}")
        onward, made = self._onward(first), []
        seats = round(STARTING * self._sold(first))
        while seats > 0:
            party = min(rng.choices(*PARTY_SIZES)[0], seats)
            seats -= party
            legs = [first]
            if onward and rng.random() < 0.3:
                for _ in range(3):
                    then = rng.choice(onward)
                    if self.room[then.id] >= party:
                        self.room[then.id] -= party
                        legs.append(then)
                        break
            known = self._known(legs)
            if known is not None and rng.random() < self.contact_share:
                made.append(self._booking(rng, legs, party, known))
        return made


def make_bookings(world: World, seed: int, contact_share: float = 0.25) -> list[Booking]:
    """The day's broken bookings whose passengers write in, in the order their messages arrive."""
    maker = _Maker(world, seed, contact_share)
    made: list[Booking] = []
    for first in sorted(world.flights.values(), key=lambda f: (f.sched_dep, f.id)):
        if DAY in first.id and not first.diverted:
            made += maker.from_flight(first)
    return sorted(made, key=lambda b: (b.sent, b.id))
