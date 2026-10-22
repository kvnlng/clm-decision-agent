"""Play the disruption coordinator: replay 19 July 2024 through the decision server, in time order.

The coordinator sends the passengers' messages through decide_many a simulated minute at a time, faster
than real time, and keeps its own copy of the world. It makes the same holds the server reports, so each
booking is labelled by the policy against the seats left when it was decided.

    uv run python examples/crowdstrike/replay_day.py examples/crowdstrike/scenario.toml [--hub ATL] [--limit N]

It starts the server itself (over stdio, with --trace), or uses one already running with --url and the key
in the variable --key-env names. The scoreboard is redrawn as the day goes; the final report and the
trace are written to runs/.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

from messages import Pool, message
from passengers import Booking, make_bookings
from policy import Label, answer
from scoreboard import Scoreboard
from world import SeatsGone, World

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


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


def clock(minute: int) -> str:
    """A simulated minute as UTC, with the day after 19 July."""
    day, rest = divmod(minute, 1440)
    return f"{rest // 60:02d}:{rest % 60:02d} UTC" + (f" on {19 + day} July" if day else "")


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
async def run(client, bookings: list[Booking], world: World, seed: int, pool: Pool, scoreboard: Scoreboard,
              cap: int, show: Callable[[str, int], None]) -> dict:
    """Send the bookings through decide_many a minute at a time; -> the scoreboard's report, with `stopped`
    saying why when the server failed part-way."""
    live = LiveLabels(world)
    stopped, now = None, "before the first message"
    for index, batch in enumerate(batches(bookings, cap)):
        labels = live.label_batch(batch)
        requests = [{"text": message(one, world, seed, pool), "context": {"booking": one.id, "now": one.sent // 60}}
                    for one in batch]
        start = time.perf_counter()
        try:
            result = await client.call_tool("decide_many", {"requests": requests})
        except Exception as error:  # pylint: disable=broad-exception-caught
            stopped = f"{type(error).__name__}: {error}"
            break
        if result.is_error:
            stopped = result.content[0].text if result.content else "the server returned an error"
            break
        replies = result.structured_content["decisions"]
        seconds = time.perf_counter() - start
        for one, reply in zip(batch, replies):
            live.apply(one, reply)
        scoreboard.add(batch, labels, replies, seconds)
        now = clock(batch[0].sent // 60)
        show(scoreboard.render(now), index)
    found = {**scoreboard.report(), "diverged": live.diverged, "clock": now}
    if stopped:
        found["stopped"] = stopped
    return found


def _show(panel: str, index: int) -> None:
    if sys.stdout.isatty():
        print("\033[H\033[J" + panel, flush=True)
    elif index % 20 == 0:
        print(panel + "\n", flush=True)


def _client(args, trace: Path):
    from mcp import Client, StdioServerParameters  # pylint: disable=import-outside-toplevel
    if args.url:
        import httpx2  # pylint: disable=import-outside-toplevel
        from mcp.client.streamable_http import streamable_http_client  # pylint: disable=import-outside-toplevel
        http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {os.environ.get(args.key_env, '')}"},
                                  timeout=600)
        return Client(streamable_http_client(args.url, http_client=http))
    return Client(StdioServerParameters(command=sys.executable, cwd=str(ROOT),
                                        args=[str(ROOT / "server.py"), str(Path(args.scenario).resolve()),
                                              "--trace", str(trace)]), read_timeout_seconds=600)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Replay the CrowdStrike day through the decision server.")
    parser.add_argument("scenario", help="path to the scenario file")
    parser.add_argument("--hub", help="only the bookings starting at this airport, such as ATL")
    parser.add_argument("--limit", type=int, help="only the first N bookings to write in")
    parser.add_argument("--batch", type=int, default=256, help="most requests in one decide_many call")
    parser.add_argument("--url", help="a running server's MCP address, instead of starting one")
    parser.add_argument("--key-env", default="DECISION_SERVER_KEY", help="variable holding the server's key")
    parser.add_argument("--name", default=time.strftime("replay-%Y%m%d-%H%M%S"), help="name of the run's files")
    args = parser.parse_args(argv)
    from scenario import ScenarioError, load_scenario  # pylint: disable=import-outside-toplevel
    try:
        seed = int(load_scenario(args.scenario).tool_settings.get("seed", 0))
    except ScenarioError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    world = World.load(seed)
    bookings = sorted(make_bookings(world, seed), key=lambda b: (b.sent, b.id))
    if args.hub:
        bookings = [one for one in bookings if one.origin == args.hub]
    bookings = bookings[:args.limit] if args.limit else bookings
    runs = HERE / "runs"
    runs.mkdir(exist_ok=True)
    trace = runs / f"{args.name}.trace.jsonl"
    print(f"Replaying {len(bookings)} bookings of seed {seed}'s day...", file=sys.stderr)

    board = Scoreboard()

    async def replay() -> dict:
        async with _client(args, trace) as client:
            return await run(client, bookings, world, seed, Pool.load(), board, args.batch, _show)

    found = {"hub": args.hub, "limit": args.limit, "seed": seed, **asyncio.run(replay())}
    (runs / f"{args.name}.json").write_text(json.dumps(found, indent=2), encoding="utf-8")
    print(f"\n{board.render(found['clock'])}\nCoordinator's world differed from the server's {found['diverged']} times")
    print(f"\nReport: {runs / f'{args.name}.json'}")
    if found.get("stopped"):
        print(f"Stopped early: {found['stopped']}", file=sys.stderr)
    if trace.exists():
        import trace_report  # pylint: disable=import-outside-toplevel
        print(trace_report.report(trace_report.summarise(trace_report.load(str(trace)))))
    return 1 if found.get("stopped") else 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
