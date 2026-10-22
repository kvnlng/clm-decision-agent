"""Build flights.csv.gz from the BTS "Reporting Carrier On-Time Performance" CSV for July 2024.

    python examples/crowdstrike/make_extract.py <the July 2024 CSV>

Keeps every flight of 19 July 2024, and the flights of 20 July scheduled to leave before noon, with
times as minutes from 19 July 00:00 UTC. BTS gives local times with no time zone; each airport's UTC
offset is worked out from the schedule itself, since a flight's scheduled arrival minus its scheduled
departure, less its scheduled flying time, is the difference between the two airports' offsets.
"""
import argparse
import csv
import gzip
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

DAYS = {"2024-07-19": 0, "2024-07-20": 1440}
NEXT_DAY_BEFORE = 1200                   # 20 July flights leaving before noon, for the overnight route
ANCHOR = ("ATL", -240)                   # Atlanta keeps Eastern Daylight Time, UTC-4
COLUMNS = ["id", "carrier", "number", "tail", "origin", "dest", "sched_dep", "sched_arr", "elapsed", "dep", "arr",
           "cancelled", "code", "diverted", "distance"]
OUT = Path(__file__).with_name("flights.csv.gz")
AIRPORTS = Path(__file__).with_name("airports.csv")


def _minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[2:])


def _wrap(minutes: int) -> int:
    """A difference of clock times, between -720 and 720."""
    return (minutes + 720) % 1440 - 720


def read(path: str) -> list[dict]:
    """The rows of the days kept."""
    with open(path, newline="", encoding="latin-1") as file:
        return [row for row in csv.DictReader(file)
                if row["FlightDate"] in DAYS and (row["FlightDate"] == "2024-07-19"
                                                  or int(row["CRSDepTime"]) < NEXT_DAY_BEFORE)]


def offsets(rows: list[dict]) -> dict[str, int]:
    """Each airport's UTC offset in minutes, spread from the anchor along the day's routes."""
    differences: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in rows:
        difference = _wrap(_minutes(row["CRSArrTime"]) - _minutes(row["CRSDepTime"])
                           - int(float(row["CRSElapsedTime"])))
        differences[row["Origin"]].append((row["Dest"], difference))
        differences[row["Dest"]].append((row["Origin"], -difference))
    found, queue = {ANCHOR[0]: ANCHOR[1]}, deque([ANCHOR[0]])
    while queue:
        airport = queue.popleft()
        votes: dict[str, Counter] = defaultdict(Counter)
        for other, difference in differences[airport]:
            if other not in found:
                votes[other][found[airport] + difference] += 1
        for other, counted in votes.items():
            if other not in found:
                found[other] = counted.most_common(1)[0][0]
                queue.append(other)
    return found


def extract(rows: list[dict], zone: dict[str, int]) -> tuple[list[dict], int]:
    """-> (the flights in the extract's columns, how many were dropped because their airports' offsets
    disagree with their schedule)."""
    flights, dropped, seen = [], 0, Counter()
    for row in rows:
        origin, dest = row["Origin"], row["Dest"]
        elapsed = int(float(row["CRSElapsedTime"]))
        start = DAYS[row["FlightDate"]]
        sched_dep = start + _minutes(row["CRSDepTime"]) - zone.get(origin, 0)
        sched_arr = start + _minutes(row["CRSArrTime"]) - zone.get(dest, 0)
        # The arrival's local clock may have passed midnight: take the arrival nearest the flying time
        sched_arr += round((sched_dep + elapsed - sched_arr) / 1440) * 1440
        if origin not in zone or dest not in zone or sched_arr - sched_dep != elapsed:
            dropped += 1
            continue
        cancelled, diverted = row["Cancelled"].startswith("1"), row["Diverted"].startswith("1")
        key = f"{row['Reporting_Airline']}{row['Flight_Number_Reporting_Airline']}-{row['FlightDate']}-{origin}"
        seen[key] += 1
        flights.append({
            "id": key if seen[key] == 1 else f"{key}-{seen[key]}",
            "carrier": row["Reporting_Airline"], "number": row["Flight_Number_Reporting_Airline"],
            "tail": row["Tail_Number"], "origin": origin, "dest": dest,
            "sched_dep": sched_dep, "sched_arr": sched_arr, "elapsed": elapsed,
            "dep": "" if cancelled or not row["DepDelay"] else sched_dep + int(float(row["DepDelay"])),
            "arr": "" if cancelled or diverted or not row["ArrDelay"] else sched_arr + int(float(row["ArrDelay"])),
            "cancelled": int(cancelled), "code": row["CancellationCode"], "diverted": int(diverted),
            "distance": int(float(row["Distance"])),
        })
    flights.sort(key=lambda flight: (flight["sched_dep"], flight["id"]))
    return flights, dropped


def write(flights: list[dict], out: Path = OUT) -> None:
    """The extract, compressed the same way every time."""
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        lines = [",".join(COLUMNS)] + [",".join(str(flight[c]) for c in COLUMNS) for flight in flights]
        packed.write(("\n".join(lines) + "\n").encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Build the CrowdStrike example's flight extract from BTS data.")
    parser.add_argument("csv", help="BTS On-Time Performance CSV for July 2024")
    args = parser.parse_args(argv)
    rows = read(args.csv)
    zone = offsets(rows)
    flights, dropped = extract(rows, zone)
    write(flights)
    used = {flight["origin"] for flight in flights} | {flight["dest"] for flight in flights}
    AIRPORTS.write_text("airport,utc_offset\n" + "".join(f"{a},{zone[a]}\n" for a in sorted(used)), encoding="utf-8")
    print(f"{len(flights)} flights written to {OUT.name}; {dropped} dropped (time zones disagree)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
