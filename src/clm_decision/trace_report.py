"""Summarise a trace written by clm-server --trace: how many calls, how fast, and where the time went."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

# Stages of a call, from the counts each line carries; "other" is what is left of the call's time
STAGES = {"waiting for CLM": "lock_wait_ms", "encode": "encode_ms", "heads": "head_ms",
          "matching corrections": "match_ms"}


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
    for name in stages[0]:
        values = np.array([stage[name] for stage in stages])
        summary["stages"][name] = {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
                                   "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
                                   "share": float(values.sum() / total)}
    span = max(r["started"] + r["ms"] / 1000 for r in records) - min(r["started"] for r in records)

    def count(key):
        return sum(record["counts"].get(key, 0) for record in records)
    # decide_many makes many decisions a call, so the rate is of decisions where they are counted
    summary["decisions"] = count("decisions") or len(records)
    summary["per_second"] = summary["decisions"] / span if span > 0 else None
    looked_up = count("cache_hits") + count("cache_misses")
    summary["cache_hit_rate"] = count("cache_hits") / looked_up if looked_up else None
    summary["texts_per_call"] = count("texts_encoded") / len(records)
    summary["tokens_per_call"] = count("tokens") / len(records)
    padded = count("padded_tokens")
    # Torch reports padding; the MLX build does not
    summary["padding"] = (padded - count("tokens")) / padded if padded else None
    return summary


def summarise(records: list[dict]) -> dict[str, dict]:
    """Per MCP tool: calls, decisions and decisions per second, each stage's time and share, cache hit rate,
    texts and tokens."""
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_tool[record["tool"]].append(record)
    return {tool: _summarise_tool(found) for tool, found in by_tool.items()}


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def report(summary: dict[str, dict]) -> str:
    """The summary as text, one table per MCP tool."""
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
        out.append(f"  cache hits {_percent(found['cache_hit_rate'])}, texts encoded per call "
                   f"{found['texts_per_call']:.1f}, tokens per call {found['tokens_per_call']:.0f}, "
                   f"padding {_percent(found['padding'])}")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Summarise a trace written by clm-server --trace.")
    parser.add_argument("trace", help="the trace file")
    args = parser.parse_args(argv)
    try:
        records = load(args.trace)
    except (OSError, ValueError) as error:
        print(f"error: cannot read {args.trace}: {error}", file=sys.stderr)
        return 2
    if not records:
        print(f"{args.trace} has no calls yet.")
        return 0
    print(report(summarise(records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
