"""Did an edit to scenario.toml, or training, make the decisions better?

    uv run clm-evaluate SCENARIO                     # replay the record of corrections
    uv run clm-evaluate SCENARIO --cases FILE        # score a file of labelled cases

A labelled case is one JSON line: {"text", "context" (optional), "route", "option" (optional, the right
option's id), "answers" (optional, {question: the right answer})}. The cases go through the server's own
steps with the scenario's tools, but no acting tool runs. Every score is per class ("41 of 50"), so a
route or an answer that is never chosen cannot hide behind the others."""
import os
# Set before huggingface_hub is imported, which reads it at import time
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# pylint: disable=wrong-import-position

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from . import clm_schema
from .clm_common import ClmError
from .dataset import reviewed_row
from .decision import decide_many, routed_text
from .record import route_verdicts, load_records, record_key
from .scenario import ScenarioError, load_scenario
from .tools import load_tools

CASES_BATCH = 32


def suggest_threshold(pairs: list[tuple[float, bool]], target: float = 0.9) -> float | None:
    """The lowest confidence threshold at which routing accuracy at or above it reaches the target."""
    for threshold in sorted({confidence for confidence, _ in pairs}):
        kept = [correct for confidence, correct in pairs if confidence >= threshold]
        if sum(kept) / len(kept) >= target:
            return threshold
    return None


def confidence_bands(pairs: list[tuple[float, bool]]) -> list[dict]:
    """Routing results grouped into five confidence bands of width 0.2."""
    counts = [[0, 0] for _ in range(5)]
    for confidence, correct in pairs:
        band = min(int(confidence * 5), 4)
        counts[band][0] += 1
        counts[band][1] += correct
    return [{"low": i / 5, "high": (i + 1) / 5, "n": n, "correct": k} for i, (n, k) in enumerate(counts)]


def match_report(route_records: list[dict], scenario, router) -> dict:
    """How many reviewed cases would match another reviewed case at the scenario's match_threshold."""
    corrected = {id(r) for r in route_verdicts(route_records, scenario)}
    report = {"n": len(route_records), "matched": 0, "same_route": 0}
    if not corrected:
        return report
    vectors = [router.embed_state(r["state_text"], scenario) for r in route_records]
    for i, case in enumerate(route_records):
        best_route, best_similarity = None, scenario.match_threshold
        for j, other in enumerate(route_records):
            if j == i or id(other) not in corrected:
                continue
            similarity = sum(a * b for a, b in zip(vectors[i], vectors[j]))
            if similarity >= best_similarity:
                best_route, best_similarity = other["human"], similarity
        if best_route is not None:
            report["matched"] += 1
            report["same_route"] += best_route == case["human"]
    return report


def _replay_routing(route_records: list[dict], scenario, router) -> dict:
    """CLM's routing of each reviewed state compared with the reviewer's route."""
    usable = [r for r in route_records if r["human"] in scenario.routes]
    per_route = {name: {"n": 0, "correct": 0} for name in scenario.routes}
    confusion, misses, pairs = Counter(), [], []
    for case in usable:
        decision = router.decide(case["state_text"], scenario)
        correct = decision["route"] == case["human"]
        per_route[case["human"]]["n"] += 1
        per_route[case["human"]]["correct"] += correct
        confusion[(case["human"], decision["route"])] += 1
        pairs.append((decision["confidence"], correct))
        if not correct:
            misses.append({"state_text": case["state_text"], "human": case["human"],
                           "clm": decision["route"], "confidence": decision["confidence"]})
    return {"n": len(usable), "correct": sum(correct for _, correct in pairs),
            "skipped": len(route_records) - len(usable), "per_route": per_route,
            "confusion": dict(confusion), "misses": misses,
            "bands": confidence_bands(pairs), "suggested_threshold": suggest_threshold(pairs)}


def _replay_row_answers(row: dict, scenario, router, answers: dict) -> None:
    """Ask the questions a reviewed row answers, grouped by the facts each hears; count into `answers`."""
    groups: dict[tuple, dict] = {}
    for name in row["answers"]:
        groups.setdefault(tuple(row["heard"].get(name, [])), {})[name] = scenario.questions[name]
    for facts, asked in groups.items():
        got = router.ask(routed_text(row["message"], list(facts)), asked)
        for name in asked:
            key = clm_schema.candidates(scenario.questions[name])[0][row["answers"][name]]
            stats = answers.setdefault(name, {}).setdefault(key, [0, 0])
            stats[0] += clm_schema.label_of(got[name]) == key
            stats[1] += 1


def _replay_answers(route_records: list[dict], scenario, router) -> tuple[dict, dict]:
    """CLM's answers and option choices on the reviewed cases, against the owner's, each question asked with
    the facts it hears when deciding: ({question: {class: [right, of]}}, {route: [right, of]})."""
    answers: dict[str, dict] = {}
    options: dict[str, list[int]] = {}
    for case in route_verdicts(route_records, scenario):
        row = reviewed_row(scenario, case)
        _replay_row_answers(row, scenario, router, answers)
        spec = scenario.routes[case["human"]].options
        if spec and row["options"]:
            ranked = router.rank(routed_text(row["message"], row["facts"]),
                                 {o["id"]: o["text"] for o in row["options"]}, spec.instructions)
            stats = options.setdefault(case["human"], [0, 0])
            stats[0] += ranked["choice"] == row["options"][row["option"]]["id"]
            stats[1] += 1
    return answers, options


def agreement(records: list[dict], scenario, router) -> dict:
    """How often CLM's route agrees with the owner's corrections: {"n", "agree"}."""
    routing = _replay_routing(records, scenario, router)
    return {"n": routing["n"], "agree": routing["correct"]}


def evaluate(records: list[dict], scenario, router) -> dict:
    """Ask CLM again about every corrected case, with live corrections off, and compare with the owner."""
    route_records = [r for r in records if r["kind"] == "route"]
    answers, options = _replay_answers(route_records, scenario, router)
    return {
        "routing": _replay_routing(route_records, scenario, router),
        "answers": answers,
        "options": options,
        # A CLM server gives no vectors, so there is nothing to measure: reuse is then by exact text
        "matching": None if getattr(router, "exact_match", False)
        else match_report([r for r in route_records if r["human"] in scenario.routes], scenario, router),
    }


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "n/a"


def _routing_lines(routing: dict, scenario) -> list[str]:
    """The routing part of the report."""
    skipped = f"; {routing['skipped']} skipped (route no longer in the scenario)" if routing["skipped"] else ""
    lines = [f"Routing: {routing['correct']} of {routing['n']} correct "
             f"({_share(routing['correct'], routing['n'])}){skipped}"]
    for name, stats in routing["per_route"].items():
        lines.append(f"  {name:<24}{stats['correct']:>4} / {stats['n']:<4} {_share(stats['correct'], stats['n'])}")

    wrong = {pair: n for pair, n in routing["confusion"].items() if pair[0] != pair[1]}
    if wrong:
        lines.append("Confusions (reviewer -> CLM):")
        for (human, clm), n in sorted(wrong.items(), key=lambda item: -item[1]):
            lines.append(f"  {human} -> {clm}  x{n}")
        lines.append("Misses:")
        for miss in routing["misses"]:
            last_line = miss["state_text"].splitlines()[-1][:80]
            lines.append(f"  [{miss['confidence']:.2f}] {miss['human']} (CLM: {miss['clm']}): {last_line}")

    lines.append("Routing accuracy by CLM confidence:")
    for band in routing["bands"]:
        if band["n"]:
            lines.append(f"  {band['low']:.1f}-{band['high']:.1f}  {band['correct']:>4} / {band['n']:<4} "
                         f"{_share(band['correct'], band['n'])}")
    suggested = routing["suggested_threshold"]
    if suggested is None:
        lines.append("No confidence_threshold reaches 90% routing accuracy on this record.")
    else:
        lines.append(f"Suggested confidence_threshold: {suggested:.2f} "
                     f"(current {scenario.confidence_threshold:.2f})")
    return lines


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


def score_cases(cases: list[dict], replies: list[dict]) -> dict:
    """Per class, (right, of): the route, the option where the route was right, and each question's answers;
    how many wrong routes were marked uncertain, so no action would have run; and the confidence threshold
    that would make the routes kept 90% right, when the replies carry a confidence."""
    found: dict = {"route": {}, "option": {}, "answers": {}, "wrong_uncertain": (0, 0)}
    pairs = []
    for case, reply in zip(cases, replies):
        right = reply["route"] == case["route"]
        _count(found["route"], case["route"], right)
        if "confidence" in reply:
            pairs.append((reply["confidence"], right))
        if not right:
            uncertain, wrong = found["wrong_uncertain"]
            found["wrong_uncertain"] = (uncertain + bool(reply["uncertain"]), wrong + 1)
        elif case.get("option") is not None:
            _count(found["option"], case["route"], (reply.get("option") or {}).get("id") == case["option"])
        for name, expected in case.get("answers", {}).items():
            if name in reply["answers"]:
                _count(found["answers"].setdefault(name, {}), expected, _value(reply["answers"][name]) == expected)
    found["threshold"] = suggest_threshold(pairs) if pairs else None
    found["bands"] = confidence_bands(pairs) if pairs else []
    return found


def _line(table: dict) -> str:
    hits = sum(right for right, _ in table.values())
    total = sum(of for _, of in table.values())
    parts = ", ".join(f"{key} {right} of {of}" for key, (right, of) in table.items())
    return f"{hits} of {total} ({hits / total:.0%}): {parts}" if total else "none"


def format_cases_report(found: dict) -> str:
    """The scores of a cases file as lines of text."""
    uncertain, wrong = found["wrong_uncertain"]
    lines = [f"Route: {_line(found['route'])}",
             f"Wrong routes marked uncertain (no action would run): {uncertain} of {wrong}",
             f"Option, where the route was right: {_line(found['option'])}"]
    lines += [f"{name}: {_line(table)}" for name, table in found["answers"].items()]
    if found.get("bands"):
        lines.append("Routing accuracy by CLM confidence:")
        lines += [f"  {band['low']:.1f}-{band['high']:.1f}  {band['correct']:>4} / {band['n']:<4} "
                  f"{_share(band['correct'], band['n'])}" for band in found["bands"] if band["n"]]
    if found.get("threshold") is not None:
        lines.append(f"Suggested confidence_threshold: {found['threshold']:.2f} (the lowest at which the routes "
                     f"kept are 90% right)")
    return "\n".join(lines)


def decide_cases(cases: list[dict], scenario, router, tools) -> list[dict]:
    """Each case decided through the server's own steps, a batch at a time. Pass a scenario without actions."""
    replies: list[dict] = []
    for start in range(0, len(cases), CASES_BATCH):
        batch = cases[start:start + CASES_BATCH]
        replies += decide_many([{"text": case["text"], **({"context": case["context"]}
                                                                   if "context" in case else {})}
                                         for case in batch], scenario, router, tools)
    return replies


def format_report(result: dict, scenario) -> str:
    """The evaluation result as text for the terminal."""
    lines = _routing_lines(result["routing"], scenario)
    if result.get("answers"):
        lines.append("Answers to the questions, by the reviewer's answer:")
        for name, classes in result["answers"].items():
            lines.append(f"  {name}")
            lines += [f"    {key:<22}{right} of {of}" for key, (right, of) in classes.items()]
    for route, (right, of) in result.get("options", {}).items():
        lines.append(f"Options of {route}: the reviewer's choice {right} of {of}")

    matching = result["matching"]
    if matching is None:
        lines.append("Reuse of reviewed routes: with a CLM server, only when the request is the same exact text.")
        return "\n".join(lines)
    lines.append(f"Reuse of reviewed routes at match_threshold {scenario.match_threshold:.2f}: "
                 f"{matching['matched']} of {matching['n']} reviewed cases match another reviewed case; "
                 f"{matching['same_route']} of those carry the route the reviewer chose.")
    return "\n".join(lines)


def load_cases(path: Path, scenario) -> list[dict]:
    """The labelled cases of a file, checked: a ScenarioError names the first bad line."""
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except ValueError as error:
            raise ScenarioError(f"{path.name} line {number}: not JSON ({error})") from error
        if not isinstance(case, dict) or not isinstance(case.get("text"), str) or "route" not in case:
            raise ScenarioError(f"{path.name} line {number}: a case needs \"text\" and \"route\"")
        if case["route"] not in scenario.routes:
            raise ScenarioError(f"{path.name} line {number}: route {case['route']!r} is not one of the scenario's "
                                f"routes")
        cases.append(case)
    if not cases:
        raise ScenarioError(f"{path.name} has no cases")
    return cases


def _main_cases(path: Path, scenario_path: str) -> int:
    try:
        scenario = replace(load_scenario(scenario_path), actions=())
        cases = load_cases(path, scenario)
        tools = load_tools(scenario)
    except (ScenarioError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        from .clm_router import make_router  # pylint: disable=import-outside-toplevel
        router, clm_note = make_router(scenario)
    except ClmError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"CLM is using {clm_note}.")
    print(format_cases_report(score_cases(cases, decide_cases(cases, scenario, router, tools))))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Replay the review record, or score a file of labelled cases, "
                                                 "against the current scenario.")
    parser.add_argument("scenario", help="path to the scenario file")
    parser.add_argument("--cases", type=Path, help="a file of labelled cases to score instead of the record")
    args = parser.parse_args(argv)
    if args.cases:
        return _main_cases(args.cases, args.scenario)
    try:
        scenario = load_scenario(args.scenario)
    except ScenarioError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        records, skipped = load_records(scenario.record_path, key=record_key(scenario))
    except (ScenarioError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not records:
        print(f"No corrections in {scenario.record_path}. "
              f"Correct some decisions with `clm-chat {args.scenario} --review` first.")
        return 0
    if skipped:
        print(f"{skipped} unreadable lines skipped in {scenario.record_path.name}")
    try:
        from .clm_router import make_router  # pylint: disable=import-outside-toplevel
        router, clm_note = make_router(scenario)
        print(f"CLM is using {clm_note}.")
        if "trained heads from" in clm_note:
            print("These heads were trained on the reviewed routes below, so routing accuracy here is not a test "
                  "of them.")
        print(format_report(evaluate(records, scenario, router), scenario))
    except ClmError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
