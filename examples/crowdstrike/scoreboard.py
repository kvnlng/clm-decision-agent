"""The scoreboard for the CrowdStrike day: what was decided, how fast, how well, and who still needs a person.

A booking goes to a person when its route is Agent, when the route is uncertain (no action runs), or when
an action failed (a seat went to someone else first). Each reason is counted; a booking with two counts
once in the total.
"""
from passengers import Booking
from policy import ROUTE_NAMES, Label


def _count(table: dict, key, right: bool) -> None:
    hits, total = table.get(key, (0, 0))
    table[key] = (hits + right, total + 1)


class Scoreboard:  # pylint: disable=too-many-instance-attributes
    """Counts as the day plays out, a batch at a time."""

    def __init__(self):
        self.bookings = self.passengers = self.automatic = self.confidently_wrong = 0
        self.seconds = 0.0
        self.route: dict = {}
        self.option: dict = {}
        self.handed = {"total": 0, "Agent route": 0, "uncertain": 0, "action failed": 0}
        self.actions: dict[str, dict[str, int]] = {}

    def add(self, bookings: list[Booking], labels: list[Label], replies: list[dict], seconds: float) -> None:
        """One batch: its bookings, their right answers, the server's replies and the time it took."""
        self.seconds += seconds
        for booking, label, reply in zip(bookings, labels, replies):
            self.bookings += 1
            self.passengers += booking.party
            right = reply["route"] == label.route
            _count(self.route, label.route, right)
            if right and label.option is not None:
                _count(self.option, label.route, (reply.get("option") or {}).get("id") == "+".join(label.option))
            if not right and not reply["uncertain"]:
                self.confidently_wrong += 1
            for action in reply["actions"]:
                counts = self.actions.setdefault(action["tool"], {"done": 0, "failed": 0, "skipped": 0})
                counts[action["status"]] = counts.get(action["status"], 0) + 1
            reasons = [reason for reason, applies in (
                ("Agent route", reply["route"] == "Agent"), ("uncertain", reply["uncertain"]),
                ("action failed", any(action["status"] == "failed" for action in reply["actions"]))) if applies]
            for reason in reasons:
                self.handed[reason] += 1
            self.handed["total"] += bool(reasons)
            self.automatic += not reasons

    def report(self) -> dict:
        """Every count, for the final report."""
        return {"bookings": self.bookings, "passengers": self.passengers, "seconds": round(self.seconds, 1),
                "per_second": round(self.bookings / self.seconds, 1) if self.seconds else 0.0,
                "route": dict(self.route), "option": dict(self.option), "automatic": self.automatic,
                "handed": dict(self.handed), "actions": {tool: dict(counts) for tool, counts in self.actions.items()},
                "confidently_wrong": self.confidently_wrong}

    def render(self, clock: str) -> str:
        """The panel for the terminal."""
        found = self.report()
        right = sum(hits for hits, _ in self.route.values())
        holds = self.actions.get("hold_seats", {})
        per_route = ", ".join(f"{name} {self.route[name][0]} of {self.route[name][1]}"
                              for name in ROUTE_NAMES if name in self.route)
        handed = self.handed
        return "\n".join([
            f"The CrowdStrike day, 19 July 2024 — {clock}",
            f"Decided: {found['bookings']} bookings, {found['passengers']} passengers, "
            f"{found['per_second']} decisions a second",
            f"Handled: automatic {found['automatic']}, to a person {handed['total']} (uncertain "
            f"{handed['uncertain']}, Agent route {handed['Agent route']}, action failed {handed['action failed']})",
            f"Seats: holds {holds.get('done', 0)} done, {holds.get('failed', 0)} failed",
            f"Right route: {right} of {found['bookings']}"
            + (f" ({right / found['bookings']:.0%})" if found["bookings"] else "")
            + f"; confidently wrong {found['confidently_wrong']}",
            f"  {per_route}",
        ])
