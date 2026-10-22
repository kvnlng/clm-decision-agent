"""Write the CrowdStrike scenario's training examples and the cases kept to check it.

    uv run python examples/crowdstrike/write_examples.py [--per-route 250] [--cases-per-route 100]

Each line is a booking's message with the right answers, and the facts and options the scenario's own
tools give for it, so training reads exactly what the server will. Training uses seed 1's day and four
fifths of the pool's lines; the check cases use seed 0's day, the demo's, and the other fifth, so CLM is
checked on wording it never saw.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import crowdstrike_tools
from messages import Pool, labels, message, split
from passengers import Booking, make_bookings
from policy import ROUTE_NAMES, answer
from world import World

HERE = Path(__file__).parent


def example(booking: Booking, world: World, pool: Pool, seed: int) -> dict:
    """One booking as an examples line. The tools must already be using `world` and the booking."""
    now = booking.sent // 60
    label = answer(booking, world, now)
    context = {"booking": booking.id, "now": now}
    request = {"text": message(booking, world, seed, pool), "route": label.route, "context": context}
    line = {"text": request["text"], "context": context, "route": label.route, "answers": labels(booking),
            "facts": {"situation": crowdstrike_tools.situation(request)["facts"]}}
    if label.option is not None:
        line.update(options=crowdstrike_tools.alternatives(request)["options"], option="+".join(label.option))
    return line


def stratified(seed: int, pool: Pool, per_route: int) -> list[dict]:
    """Up to `per_route` examples of each route from seed's day, so rare routes are not drowned out."""
    world = World.load(seed)
    bookings = make_bookings(world, seed)
    crowdstrike_tools.use(world, {one.id: one for one in bookings})
    random.Random(seed).shuffle(bookings)
    found: dict[str, list[dict]] = {name: [] for name in ROUTE_NAMES}
    for booking in bookings:
        route = answer(booking, world, booking.sent // 60).route
        if len(found[route]) < per_route:
            found[route].append(example(booking, world, pool, seed))
            if all(len(lines) == per_route for lines in found.values()):
                break
    return [line for name in ROUTE_NAMES for line in found[name]]


def write(path: Path, lines: list[dict]) -> None:
    """The lines as JSON lines, and how many of each route."""
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    counts = Counter(line["route"] for line in lines)
    print(f"{path.name}: {len(lines)} lines, " + ", ".join(f"{name} {counts[name]}" for name in ROUTE_NAMES))


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Write the CrowdStrike scenario's examples and check cases.")
    parser.add_argument("--per-route", type=int, default=250)
    parser.add_argument("--cases-per-route", type=int, default=100)
    args = parser.parse_args(argv)
    seen, unseen = split(Pool.load())
    write(HERE / "scenario.examples.jsonl", stratified(1, seen, args.per_route))
    write(HERE / "check.cases.jsonl", stratified(0, unseen, args.cases_per_route))
    return 0


if __name__ == "__main__":
    sys.exit(main())
