"""Play the disruption coordinator: replay 19 July 2024 through the decision server, in time order.

The coordinator sends the passengers' messages through decide_many a simulated minute at a time, faster
than real time, and keeps its own copy of the world. It makes the same holds the server reports, so each
booking is labelled by the policy against the seats left when it was decided.
"""
from passengers import Booking
from policy import Label, answer
from world import SeatsGone, World


def batches(bookings: list[Booking], cap: int) -> list[list[Booking]]:
    """The bookings in the order they wrote, a minute at a time, no batch larger than `cap`."""
    found: list[list[Booking]] = []
    minute = None
    for one in sorted(bookings, key=lambda b: (b.sent, b.id)):
        if one.sent // 60 != minute or len(found[-1]) == cap:
            found.append([])
            minute = one.sent // 60
        found[-1].append(one)
    return found


class LiveLabels:
    """The policy's answers against the seats left. The server finds every request's facts and options
    before any of the batch's actions, so a batch is labelled before its holds are made."""

    def __init__(self, world: World):
        self.world = world
        self.diverged = 0        # holds the server made that this world could not: the two worlds differ

    def label_batch(self, bookings: list[Booking]) -> list[Label]:
        """The right answer for each booking of a batch, at the minute it wrote."""
        return [answer(one, self.world, one.sent // 60) for one in bookings]

    def apply(self, booking: Booking, reply: dict) -> None:
        """Make the hold the server made for this booking, if it made one."""
        if not any(action["tool"] == "hold_seats" and action["status"] == "done" for action in reply["actions"]):
            return
        try:
            self.world.hold(booking.id, tuple(reply["option"]["id"].split("+")), booking.party)
        except (SeatsGone, ValueError):
            self.diverged += 1
