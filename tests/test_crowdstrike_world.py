"""Tests for the CrowdStrike example's world: the flight extract, capacities, search and seats."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import csv
import gzip
from pathlib import Path

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
