"""How well the scenario decides on the check cases: messages whose wording training never saw.

    uv run python examples/crowdstrike/check.py examples/crowdstrike/scenario.toml [--cases check.cases.jsonl]

Every case goes through the decision server's own steps (decision.decide_many), with the tools on the
demo's day, but with no actions: a hold would change the seats the later cases' answers were worked out on.
Each result is per class ("41 of 50"), so a route that is never chosen cannot hide behind the others.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# pylint: disable=wrong-import-position
import decision  # noqa: E402
from scenario import ScenarioError, load_scenario  # noqa: E402
from tools import load_tools  # noqa: E402

BATCH = 32


def _count(table: dict, key, right: bool) -> None:
    hits, total = table.get(key, (0, 0))
    table[key] = (hits + right, total + 1)


def _value(answer: dict):
    """A reply's answer as the cases write it: true or false, a level, or a choice's key."""
    kind = answer["type"]
    if kind == "noul":
        return answer["noul"] >= 0.5
    if kind == "score":
        return round(answer["score"])
    return answer["choice"]


def score(cases: list[dict], replies: list[dict]) -> dict:
    """Per class, (right, of): the route, the option where the route was right, and each question's answers;
    and how many wrong routes were marked uncertain, so no action would have run."""
    found: dict = {"route": {}, "option": {}, "answers": {}, "wrong_uncertain": (0, 0)}
    for case, reply in zip(cases, replies):
        right = reply["route"] == case["route"]
        _count(found["route"], case["route"], right)
        if not right:
            uncertain, wrong = found["wrong_uncertain"]
            found["wrong_uncertain"] = (uncertain + bool(reply["uncertain"]), wrong + 1)
        elif case.get("option") is not None:
            _count(found["option"], case["route"], (reply.get("option") or {}).get("id") == case["option"])
        for name, expected in case.get("answers", {}).items():
            if name in reply["answers"]:
                _count(found["answers"].setdefault(name, {}), expected, _value(reply["answers"][name]) == expected)
    return found


def _line(table: dict) -> str:
    hits = sum(right for right, _ in table.values())
    total = sum(of for _, of in table.values())
    parts = ", ".join(f"{key} {right} of {of}" for key, (right, of) in table.items())
    return f"{hits} of {total} ({hits / total:.0%}): {parts}" if total else "none"


def report(found: dict) -> str:
    """The scores as lines of text."""
    uncertain, wrong = found["wrong_uncertain"]
    lines = [f"Route: {_line(found['route'])}",
             f"Wrong routes marked uncertain (no action would run): {uncertain} of {wrong}",
             f"Option, where the route was right: {_line(found['option'])}"]
    lines += [f"{name}: {_line(table)}" for name, table in found["answers"].items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Score the CrowdStrike scenario on its check cases.")
    parser.add_argument("scenario", help="path to the scenario file")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("check.cases.jsonl"))
    args = parser.parse_args(argv)
    try:
        scenario = replace(load_scenario(args.scenario), actions=())
        tools = load_tools(scenario)
    except ScenarioError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line]
    from clm_router import ClmError, make_router  # pylint: disable=import-outside-toplevel
    try:
        router, note = make_router(scenario)
    except ClmError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"CLM is using {note}.")
    replies = []
    for start in range(0, len(cases), BATCH):
        batch = cases[start:start + BATCH]
        replies += decision.decide_many([{"text": case["text"], "context": case["context"]} for case in batch],
                                        scenario, router, tools)
    print(report(score(cases, replies)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
