"""The check_call story: an agent's own AI proposes both a rebooking and a refund. Does CLM flag the wrong one?

    uv run python examples/crowdstrike/check_calls.py examples/crowdstrike/scenario.toml [--cases check.cases.jsonl]

For each check case whose right route is Rebook or Refund, the agent proposes two calls, holding seats on a
new flight and refunding the fare, and asks check_call about each, as an agent would over MCP: the message
and the proposed call, nothing else. A case is caught when CLM approves the right call and refuses the
wrong one; the right call is preferred when its probability is the higher of the two. Results go to
runs/check-calls.json, which git ignores.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

# pylint: disable=wrong-import-position
import decision  # noqa: E402
from scenario import ScenarioError, load_scenario  # noqa: E402

# Each call says what it is for, in the words an agent's tool description would use
PROPOSALS = {"hold_seats": "hold seats for the party on a new flight, to rebook them",
             "issue_refund": "refund the fare; the passenger will not travel"}
RIGHT = {"Rebook": "hold_seats", "Refund": "issue_refund"}


def check(cases: list[dict], scenario, router) -> dict:
    """Per right route: how many cases, how often each proposed call was approved, how many were caught, and
    how often the right call had the higher probability."""
    found: dict = {"cases": {}, "approved": {}, "caught": {}, "preferred": {}, "p": []}
    for case in cases:
        if case["route"] not in RIGHT:
            continue
        approved, p = {}, {}
        for tool, purpose in PROPOSALS.items():
            call = {"tool": tool, "arguments": {"booking": (case.get("context") or {}).get("booking")},
                    "purpose": purpose}
            result = decision.check_call(case["text"], call, scenario, router)
            approved[tool], p[tool] = result["approved"], result["p"]
            found["p"].append({"route": case["route"], "tool": tool, "p": round(result["p"], 3)})
        route = case["route"]
        found["cases"][route] = found["cases"].get(route, 0) + 1
        counts = found["approved"].setdefault(route, dict.fromkeys(PROPOSALS, 0))
        for tool, yes in approved.items():
            counts[tool] += yes
        wrong = next(tool for tool in PROPOSALS if tool != RIGHT[route])
        found["caught"][route] = found["caught"].get(route, 0) + (approved[RIGHT[route]] and not approved[wrong])
        found["preferred"][route] = found["preferred"].get(route, 0) + (p[RIGHT[route]] > p[wrong])
    return found


def report(found: dict) -> str:
    """The counts as lines of text."""
    lines = []
    for route, count in found["cases"].items():
        approved = ", ".join(f"{tool} approved {n} of {count}" for tool, n in found["approved"][route].items())
        lines.append(f"{route} ({count} cases): {approved}; caught {found['caught'][route]} of {count}; "
                     f"the right call preferred {found['preferred'][route]} of {count}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Ask check_call about a rebooking and a refund for each case.")
    parser.add_argument("scenario", help="path to the scenario file")
    parser.add_argument("--cases", type=Path, default=HERE / "check.cases.jsonl")
    args = parser.parse_args(argv)
    try:
        scenario = load_scenario(args.scenario)
        cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (ScenarioError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    from clm_router import ClmError, make_router  # pylint: disable=import-outside-toplevel
    try:
        router, note = make_router(scenario)
    except ClmError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"CLM is using {note}; approved at p >= {scenario.tool_check_threshold}.")
    found = check(cases, scenario, router)
    print(report(found))
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "check-calls.json").write_text(json.dumps(found, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
