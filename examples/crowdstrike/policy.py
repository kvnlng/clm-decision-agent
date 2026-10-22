"""This demo's disruption policy, as policy.md writes it: the right answer for each booking.

The same file gives the route descriptions the decision server reads, so what CLM is asked and what
counts as right come from one document.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from passengers import Booking
from world import CONNECTION, PARTNERS, Journey, World, carriers_of, group

ROUTE_NAMES = ("Rebook", "Rebook_Partner", "Overnight", "Refund", "Agent", "No_Action")
ON_TIME = 60                 # minutes late that still count as arriving as planned


def _routes(text: str) -> dict[str, str]:
    found = {}
    for section in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        title, _, body = section.partition("\n")
        if title.strip() in ROUTE_NAMES:
            found[title.strip()] = " ".join(body.split())
    return {name: found[name] for name in ROUTE_NAMES}


ROUTES = _routes((Path(__file__).parent / "policy.md").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Label:
    """The right answer: a route, and for the rebooking routes the journey to book."""
    route: str
    option: tuple[str, ...] | None


def _kept(booking: Booking, world: World) -> int | None:
    """When the booked journey now arrives, or None when it no longer gets them there."""
    legs = [world.flights[leg] for leg in booking.legs]
    if any(leg.cancelled or leg.diverted for leg in legs):
        return None
    if len(legs) == 2 and legs[1].leaves - legs[0].lands < CONNECTION:
        return None
    return legs[-1].lands


def _search(world: World, booking: Booking, now: int, carriers: set[str],  # pylint: disable=too-many-arguments
            by: int | None, *, day: int, limit: int) -> tuple[Journey, ...]:
    """Up to `limit` journeys leaving on `day` (0 for 19 July, 1 for the 20th, before noon) from where the
    passenger is, earliest arrival first, arriving by `by` if given."""
    first = world.flights[booking.legs[0]]
    airport, after = booking.origin, now
    if not first.cancelled and now >= first.leaves:
        # They are on board: only what comes after it can change, with time to make the connection
        airport, after = first.dest, first.lands + CONNECTION
    midnight = 1440 - world.offsets.get(airport, 0)

    def today(journey: Journey) -> bool:
        # Every flight leaves today where it leaves from: a connection overnight is a hotel night
        return all(world.flights[leg].leaves < 1440 - world.offsets.get(world.flights[leg].origin, 0)
                   for leg in journey.legs)

    def tomorrow_morning(journey: Journey) -> bool:
        return journey.dep < midnight + 720

    if day == 1:
        after = max(after, midnight)
    return tuple(world.search(airport, booking.dest, after, now, booking.party, carriers, deadline=by, limit=limit,
                              accept=tomorrow_morning if day == 1 else today))


@dataclass(frozen=True)
class Situation:
    """What the day left a booking with, from its record alone: how its own flights now go, and the
    alternatives, each earliest arrival first. The searches are skipped when the flights are on plan."""
    kept: int | None
    on_plan: bool
    own: tuple[Journey, ...] = ()
    partner: tuple[Journey, ...] = ()
    tomorrow: tuple[Journey, ...] = ()


def situation(booking: Booking, world: World, now: int, limit: int = 3) -> Situation:
    """The booking's situation at minute `now`. It reads only the record (legs, party, fare), never what
    the passenger alone knows, so the tools may state it."""
    kept = _kept(booking, world)
    if kept is not None and kept <= world.flights[booking.legs[-1]].sched_arr + ON_TIME:
        return Situation(kept, True)
    # A new journey has to beat the one they have, if it still gets them there
    by = None if kept is None else kept - 1
    airline = group(world.flights[booking.legs[0]].carrier)
    own = _search(world, booking, now, carriers_of(airline), by, day=0, limit=limit)
    partner: tuple[Journey, ...] = ()
    partners = set().union(*(carriers_of(group(code)) for code in PARTNERS.get(airline, ())))
    if booking.fare.partner_ok and partners:
        partner = _search(world, booking, now, partners, by, day=0, limit=limit)
    tomorrow = _search(world, booking, now, carriers_of(airline), None, day=1, limit=limit)
    return Situation(kept, False, own, partner, tomorrow)


def _meets(journeys: tuple[Journey, ...], firm: int | None) -> Journey | None:
    """The first journey, if it arrives by the firm deadline: none after it arrives sooner."""
    if journeys and (firm is None or journeys[0].arr <= firm):
        return journeys[0]
    return None


def answer(booking: Booking, world: World, now: int) -> Label:  # pylint: disable=too-many-return-statements
    """The policy's answer for a booking whose passenger writes in at minute `now`."""
    if booking.vague:
        return Label("Agent", None)
    if not booking.still_travelling:
        return Label("Refund", None)
    found = situation(booking, world, now)
    if found.on_plan:
        return Label("No_Action", None)
    firm = booking.deadline if booking.deadline_kind == "firm" else None
    if journey := _meets(found.own, firm):
        return Label("Rebook", journey.legs)
    if journey := _meets(found.partner, firm):
        return Label("Rebook_Partner", journey.legs)
    if found.kept is not None and (firm is None or found.kept <= firm):
        return Label("No_Action", None)
    if journey := _meets(found.tomorrow, firm):
        return Label("Overnight", journey.legs)
    if firm is not None:
        return Label("Refund" if booking.fare.refundable else "Agent", None)
    return Label("Agent", None)
