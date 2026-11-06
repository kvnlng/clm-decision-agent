"""Summarise a trace written by clm-server --trace: how many calls, how fast, and where the time went."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

# Stages of a call, from the counts each line carries; "other" is what is left of the call's time
STAGES = {"waiting for CLM": "lock_wait_ms", "encode": "encode_ms", "heads": "head_ms",
          "matching corrections": "match_ms", "queries": "query_ms", "rules": "decision_ms",
          "calculations": "calculation_ms", "judges": "judgement_ms"}
# An environment's steps, counted per call
STEPS = {"queries": "query", "decisions": "decision", "calculations": "calculation", "judgements": "judgement"}


def load(path: str) -> list[dict]:
    """The trace's lines, one dict per call. A last line cut off mid-write, when the server was stopped, is left
    out."""
    with open(path, encoding="utf-8") as file:
        lines = [line for line in file if line.strip()]
    records = [json.loads(line) for line in lines[:-1]]
    try:
        records.extend(json.loads(line) for line in lines[-1:])
    except ValueError:
        pass
    return records


def parse_time(text: str, now: float) -> float:
    """An epoch time from `15m`, `2h` or `30s` (that long before `now`) or an ISO time such as
    2026-11-06T12:00:00+00:00. A time with no offset is local. Anything else raises ValueError."""
    ago = re.fullmatch(r"(\d+(?:\.\d+)?)([smh])", text.strip())
    if ago:
        return now - float(ago.group(1)) * {"s": 1, "m": 60, "h": 3600}[ago.group(2)]
    try:
        return datetime.fromisoformat(text.strip()).timestamp()
    except ValueError:
        raise ValueError(f"{text!r} is neither a time such as 15m, 2h or 30s nor an ISO time") from None


def window(records: list[dict], since: float | None, until: float | None) -> list[dict]:
    """The calls that started from `since` to `until`, both ends included and either open, without the
    health-check calls the server makes itself."""
    return [record for record in records if not record.get("synthetic")
            and (since is None or record["started"] >= since) and (until is None or record["started"] <= until)]


def _caller(record: dict) -> str:
    caller = record.get("caller") or {}
    return caller.get("client_id") or caller.get("key_id") or "unknown"


def _stages(record: dict) -> dict[str, float]:
    counts = record["counts"]
    stages = {name: float(counts.get(key, 0)) for name, key in STAGES.items()}
    for kind, name in (("inform", "inform tools"), ("act", "acting tools")):
        stages[name] = sum(tool["ms"] for tool in record["tools"] if tool["kind"] == kind)
    stages["other"] = max(record["ms"] - sum(stages.values()), 0.0)
    return {"total": record["ms"], **stages}


def _summarise_tool(records: list[dict]) -> dict:
    stages = [_stages(record) for record in records]
    total = sum(stage["total"] for stage in stages) or 1.0
    summary: dict = {"calls": len(records), "stages": {}}
    # A stage no call had, such as CLM's in an environment, is left out
    for name in [name for name in stages[0] if name in ("total", "other") or any(stage[name] for stage in stages)]:
        values = np.array([stage[name] for stage in stages])
        summary["stages"][name] = {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
                                   "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
                                   "share": float(values.sum() / total)}
    span = max(r["started"] + r["ms"] / 1000 for r in records) - min(r["started"] for r in records)

    def count(key):
        return sum(record["counts"].get(key, 0) for record in records)
    # decide_many makes many decisions a call, so the rate is of decisions where they are counted
    summary["decisions"] = count("decisions") or len(records)
    # One call of one decision has no rate: its span is only its own time
    timed = span > 0 and (len(records) > 1 or summary["decisions"] > 1)
    summary["per_second"] = summary["decisions"] / span if timed else None
    looked_up = count("cache_hits") + count("cache_misses")
    summary["cache_hit_rate"] = count("cache_hits") / looked_up if looked_up else None
    summary["texts_per_call"] = count("texts_encoded") / len(records)
    summary["tokens_per_call"] = count("tokens") / len(records)
    padded = count("padded_tokens")
    # Torch reports padding; the MLX build does not
    summary["padding"] = (padded - count("tokens")) / padded if padded else None
    summary["steps_per_call"] = {name: count(key) / len(records) for name, key in STEPS.items() if count(key)}
    return summary


def summarise(records: list[dict]) -> dict[str, dict]:
    """Per MCP tool: calls, decisions and decisions per second, each stage's time and share, cache hit rate,
    texts and tokens, and for an environment its steps per call."""
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_tool[record["tool"]].append(record)
    return {tool: _summarise_tool(found) for tool, found in by_tool.items()}


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def report(summary: dict[str, dict], records: list[dict] | None = None) -> str:
    """The summary as text, one table per MCP tool. Given the records it summarises, it ends with a table of
    calls and failures per caller (a client id, else a shared key's id, else unknown) and the failure count."""
    out = []
    for tool, found in summary.items():
        rate = "" if found["per_second"] is None else f", {found['per_second']:.1f} a second"
        made = "" if found["decisions"] == found["calls"] else f", {found['decisions']} decisions"
        out.append(f"{tool}: {found['calls']} calls{made}{rate}")
        out.append(f"  {'stage':<22}{'mean ms':>10}{'p50':>10}{'p95':>10}{'p99':>10}{'share':>8}")
        for name, stage in found["stages"].items():
            share = "" if name == "total" else _percent(stage["share"])
            out.append(f"  {name:<22}{stage['mean']:>10.1f}{stage['p50']:>10.1f}{stage['p95']:>10.1f}"
                       f"{stage['p99']:>10.1f}{share:>8}")
        if found["texts_per_call"] or found["cache_hit_rate"] is not None:
            out.append(f"  cache hits {_percent(found['cache_hit_rate'])}, texts encoded per call "
                       f"{found['texts_per_call']:.1f}, tokens per call {found['tokens_per_call']:.0f}, "
                       f"padding {_percent(found['padding'])}")
        if found["steps_per_call"]:
            out.append("  steps per call: " + ", ".join(f"{value:.1f} {name}"
                                                         for name, value in found["steps_per_call"].items()))
        out.append("")
    if records:
        calls = Counter(_caller(record) for record in records)
        failed = Counter(_caller(record) for record in records if "error" in record["outcome"])
        out.append(f"  {'caller':<22}{'calls':>10}{'errors':>10}")
        out.extend(f"  {name:<22}{count:>10}{failed[name]:>10}" for name, count in calls.most_common())
        out.append(f"errors: {sum(failed.values())} of {len(records)} calls")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Summarise a trace written by clm-server --trace.")
    parser.add_argument("trace", help="the trace file")
    parser.add_argument("--since", help="only calls from then: 15m, 2h or 30s ago, or an ISO time")
    parser.add_argument("--until", help="only calls up to then, written the same way")
    args = parser.parse_args(argv)
    now = time.time()
    try:
        since = None if args.since is None else parse_time(args.since, now)
        until = None if args.until is None else parse_time(args.until, now)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        records = load(args.trace)
    except (OSError, ValueError) as error:
        print(f"error: cannot read {args.trace}: {error}", file=sys.stderr)
        return 2
    if not records:
        print(f"{args.trace} has no calls yet.")
        return 0
    records = window(records, since, until)
    if not records:
        def shown(moment):
            return datetime.fromtimestamp(moment).astimezone().isoformat(" ", "seconds")
        start = "the start" if since is None else shown(since)
        print(f"no calls between {start} and {shown(now if until is None else until)}.")
        return 0
    print(report(summarise(records), records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
