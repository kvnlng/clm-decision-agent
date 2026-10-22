"""Tools for the CrowdStrike example. The decision server calls each one with the request, whose context
is {"booking": "<id>", "now": <minute>}; without "now", the moment the passenger wrote.

    situation     an informing tool the route hears: the booking, what happened, the fare and what is possible
    alternatives  the rebooking routes' options: the soonest journeys of the route's kind, with seats
    hold_seats    an acting tool: holds the whole party's seats on the chosen journey, or fails naming a full flight
    book_hotel    an acting tool: a night's hotel near where they are (a mock)
    issue_refund  an acting tool: refunds the fare (a mock)
    draft_reply   an acting tool: the message back to the passenger

The tools work things out and CLM judges: every fact is a plain sentence with the times already compared.
They read only what a booking record holds (flights, party, fare), never what the passenger alone knows;
a time the passenger must arrive by reaches them only as words in the message.
"""
from __future__ import annotations

import re
from pathlib import Path

from passengers import Booking, make_bookings
from policy import Situation, situation as situation_of
from world import PARTNERS, Journey, World, group

STATE: dict = {"world": None, "bookings": {}}
# A clock time as World.local writes it: "18:05", or "09:00 on 20 July"
TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?: on (\d{1,2}) july)?\b", re.IGNORECASE)


def use(world: World, bookings: dict[str, Booking]) -> None:
    """Decide on this world and these bookings."""
    STATE.update(world=world, bookings=bookings)


def configure(settings: dict, scenario_dir: Path) -> None:  # pylint: disable=unused-argument
    """The day's world and its passengers, from the `seed` setting (0 unless given)."""
    seed = int(settings.get("seed", 0))
    world = World.load(seed)
    use(world, {one.id: one for one in make_bookings(world, seed)})


def _booking(request: dict) -> tuple[Booking, int]:
    """The request's booking, and the moment of the decision."""
    context = request.get("context") or {}
    found = STATE["bookings"].get(context.get("booking"))
    if found is None:
        raise ValueError(f"context has no known booking: {context.get('booking')!r}")
    return found, int(context.get("now", found.sent // 60))


def _deadlines(text: str, booking: Booking, world: World) -> list[tuple[str, int]]:
    """Each time the message mentions, as written, with its minute (UTC) taken as the destination's clock.
    The booked departure's time is left out: it is when they were to leave, not a time to arrive by."""
    departure = world.local(booking.origin, world.flights[booking.legs[0]].sched_dep).lower()
    found = []
    for match in TIME.finditer(text):
        hours, minutes, day = int(match[1]), int(match[2]), int(match[3] or 19)
        if match[0].lower() == departure or hours > 23 or minutes > 59:
            continue
        minute = (day - 19) * 1440 + hours * 60 + minutes - world.offsets.get(booking.dest, 0)
        found.append((match[0], minute))
    return found


def _span(minutes: int) -> str:
    hours, rest = divmod(minutes, 60)
    return " ".join(part for part in (f"{hours} h" if hours else "", f"{rest} min" if rest else "") if part)


def _lands(minute: int, booking: Booking, world: World, deadlines: list[tuple[str, int]],
           verb: str = "lands") -> str:
    """'lands at 13:40', with how that compares to each time the message mentions."""
    words = f"{verb} at {world.local(booking.dest, minute)}"
    for written, by in deadlines:
        words += f", {'before' if minute <= by else 'after'} the {written} they mention"
    return words


def _happened(booking: Booking, world: World, found: Situation, deadlines: list[tuple[str, int]]) -> list[str]:
    flights = [world.flights[leg] for leg in booking.legs]
    broken = [f"{f.carrier}{f.number} is cancelled." if f.cancelled else f"{f.carrier}{f.number} was diverted."
              for f in flights if f.cancelled or f.diverted]
    if broken:
        return broken
    if found.kept is None:
        first, second = flights
        return [f"{first.carrier}{first.number} now lands at {first.dest} at {world.local(first.dest, first.lands)}, "
                f"too late for the connection to {second.carrier}{second.number}."]
    if found.on_plan:
        return ["Their flights still go and land within an hour of plan."]
    late = found.kept - flights[-1].sched_arr
    landing = _lands(found.kept, booking, world, deadlines, "land")
    return [f"Their flights still go but now {landing}, {_span(late)} late."]


def _alternatives(booking: Booking, world: World, found: Situation, deadlines: list[tuple[str, int]]) -> list[str]:
    seats = "with seats for all of them"
    sooner = ", sooner than their flights now will" if found.kept is not None else ""
    there = "gets them there sooner" if found.kept is not None else "gets them there"

    def soonest(journeys: tuple[Journey, ...], whose: str, none: str) -> str:
        if journeys:
            return f"The soonest {whose} {seats} {_lands(journeys[0].arr, booking, world, deadlines)}{sooner}."
        return f"{none} {seats} {there}."

    facts = [soonest(found.own, "of our own flights today", "None of our own flights today")]
    if found.own:
        return facts
    airline = group(world.flights[booking.legs[0]].carrier)
    if booking.fare.partner_ok and PARTNERS.get(airline):
        facts.append(soonest(found.partner, "partner flight today", "No partner flight today"))
        if found.partner:
            return facts
    if found.tomorrow:
        facts.append(f"Our first flight tomorrow morning {seats} "
                     f"{_lands(found.tomorrow[0].arr, booking, world, deadlines)}.")
    else:
        facts.append("None of our flights tomorrow morning has seats for all of them.")
    return facts


def situation(request: dict) -> dict:
    """The booking, what happened to it, the fare, and the soonest journeys still possible."""
    booking, now = _booking(request)
    world = STATE["world"]
    found = situation_of(booking, world, now)
    deadlines = _deadlines(request.get("text", ""), booking, world)
    flights = [world.flights[leg] for leg in booking.legs]
    route = ", then ".join(f"{f.carrier}{f.number} from {f.origin} at {world.local(f.origin, f.sched_dep)}"
                           for f in flights)
    people = "1 person" if booking.party == 1 else f"{booking.party} people"
    facts = [f"The booking, for {people}: {route} to {booking.dest}, due to land "
             f"{world.local(booking.dest, flights[-1].sched_arr)} (local times)."]
    facts += _happened(booking, world, found, deadlines)
    facts.append(f"The fare is {'' if booking.fare.refundable else 'not '}refundable and "
                 f"{'allows' if booking.fare.partner_ok else 'does not allow'} partner airlines.")
    if not found.on_plan:
        facts += _alternatives(booking, world, found, deadlines)
    return {"facts": facts}


KINDS = {"Rebook": "own", "Rebook_Partner": "partner", "Overnight": "tomorrow"}


def alternatives(request: dict) -> dict:
    """The journeys the route could book, soonest first, each said in plain words."""
    booking, now = _booking(request)
    world = STATE["world"]
    found = situation_of(booking, world, now)
    journeys = getattr(found, KINDS[request["route"]]) if request.get("route") in KINDS else ()
    options = []
    for index, journey in enumerate(journeys):
        text = journey.text(world) + ("; lands first of these" if index == 0 else "")
        if found.kept is not None and journey.arr < found.kept:
            text += f"; {_span(found.kept - journey.arr)} sooner than their flights now"
        options.append({"id": "+".join(journey.legs), "text": text})
    return {"options": options}


def _legs(request: dict) -> tuple[str, ...]:
    return tuple(request["option"]["id"].split("+"))


def _held(request: dict) -> tuple[str, ...]:
    """The chosen journey, once hold_seats has held it: the acting tools run in turn, whatever came before."""
    booking, _ = _booking(request)
    legs = _legs(request)
    if STATE["world"].holds.get(booking.id, ((), 0))[0] != legs:
        raise ValueError(f"no seats are held on {request['option']['id']} for {booking.id}")
    return legs


def hold_seats(request: dict) -> dict:
    """Hold the whole party's seats on every flight of the chosen journey; SeatsGone names a full one."""
    booking, _ = _booking(request)
    legs = _legs(request)
    STATE["world"].hold(booking.id, legs, booking.party)
    return {"held": list(legs)}


def book_hotel(request: dict) -> dict:
    """A night's hotel near the airport they leave from in the morning (this demo books nothing real)."""
    first = STATE["world"].flights[_held(request)[0]]
    return {"hotel": f"near {first.origin}", "nights": 1}


def issue_refund(request: dict) -> dict:
    """Refund the fare (this demo pays nothing)."""
    booking, _ = _booking(request)
    return {"refunded": booking.id}


def draft_reply(request: dict) -> dict:
    """The message back to the passenger, from the route and the chosen journey."""
    world = STATE["world"]
    journey = ""
    if request.get("option"):
        legs = _held(request)
        flights = [world.flights[leg] for leg in legs]
        journey = Journey(legs, flights[0].leaves, flights[-1].lands).text(world)
    replies = {
        "Rebook": f"We have booked you on {journey}.",
        "Rebook_Partner": f"We have booked you with our partner airline on {journey}.",
        "Overnight": f"We have booked a hotel for tonight, and you on {journey} tomorrow morning.",
        "Refund": "We have refunded your fare.",
        "No_Action": "Your flights still go, so you need do nothing; we are sorry for the delay.",
    }
    return {"reply": replies.get(request.get("route"), "A member of our team will contact you.")}


TOOLS = [situation, alternatives, hold_seats, book_hotel, issue_refund, draft_reply]
