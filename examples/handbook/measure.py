"""Measure the five questions in six setups, on Claude and on a local model: with no tools, with the rules only,
with the rules and the handbook, with the handbook only, and with rules derived from the handbook by Claude or by the
local model.

    uv run python examples/handbook/measure.py derive claude|ollama
    uv run python examples/handbook/measure.py restore claude|ollama
    uv run python examples/handbook/measure.py run [--backends claude,ollama] [--questions q1,...] [--setups none,...]
    uv run python examples/handbook/measure.py compare
    uv run python examples/handbook/measure.py score

derive runs `clm-derive run --fresh` on environment_derived.toml with that backend, approves every entry that passed
its check, labelled as a scripted approval and not a person's, and keeps a copy of the layer in
runs/measure/derived-BACKEND/; restore puts that copy back. compare evaluates each kept layer against the second
key, the hand-written rules/*.dmn and graph/nodes.csv, and writes runs/measure/second_key.json: above all, the
entries that passed the check and are wrong, which a person approving them would have had to catch.

run asks clm-agent each question in each setup, on each backend, and writes each answer to runs/measure/, which git
ignores: `none` with --no-tools, `rules` with environment_without_documents.toml, `documents` with environment.toml,
`documents_only` with environment_documents_only.toml, and `derived` and `derived_local` with
environment_derived.toml, which run only when the layer in derived/ is Claude's or gemma4's. score reads them
against answer_key.toml:

- figures: Claude (claude-opus-5-5) reads, for each figure the key lists, the value the answer gives for it and
  whether it is stated as fact or hedged, and any further figure stated about Meridian. It is shown what each figure
  is about, never its value. This script compares: an item is right within 0.5% of the key, wrong otherwise (also
  counted as superseded when it is a 2024 value), not given when absent. A hedged value is counted on its own line,
  neither right nor wrong. Wrong as fact, the headline, is the wrong items. Further figures stated as fact are not
  graded, since the key does not hold them: they are counted as from a tool when a tool returned them
  (number_check.py), and as unsourced otherwise.
- verdict: read by Claude from a fixed list for each question, as in noncompete.
- policy cited: how many of the key's handbook sections the answer names by section number ("§ 4").

The company, its people and its policies are made up. Not HR, tax or legal advice.
"""
import argparse
import csv
import itertools
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

from clm_decision import decision_agent, derive, derived
from clm_decision.dmn import DmnError, evaluate, load_table
from clm_decision.number_check import unsupported

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "measure"
DERIVED = HERE / "environment_derived.toml"
SETUPS = {"none": (HERE / "environment.toml", ["--no-tools"]),
          "rules": (HERE / "environment_without_documents.toml", []),
          "documents": (HERE / "environment.toml", []),
          "documents_only": (HERE / "environment_documents_only.toml", []),
          "derived": (DERIVED, []),
          "derived_local": (DERIVED, [])}
# Which model's layer each derived setup needs in derived/
LAYER_MODELS = {"derived": decision_agent.MODEL, "derived_local": decision_agent.LOCAL_MODEL}
SCRIPTED = "measure.py: scripted approval of every checked entry, not a person"
AS_OF = "2026-12-01"
# The second key for the match's tiers, which no table or graph holds: handbook_functions.company_match, written by
# hand from § 4(3) (100% of the first 3% of pay, 50% of the next 3%); graph/nodes.csv holds the other two figures
HAND_TIERS = {"first_tier_percent": 3, "first_tier_rate": 100, "second_tier_percent": 3, "second_tier_rate": 50}
READER = "claude-opus-5-5"
NOT_GIVEN = "not given"
CLOSE = 0.005


def _key() -> dict:
    return tomllib.loads((HERE / "answer_key.toml").read_text(encoding="utf-8"))


def _handbook() -> dict[str, dict]:
    lines = (HERE / "handbook.jsonl").read_text(encoding="utf-8").splitlines()
    return {document["id"]: document for document in map(json.loads, lines)}


def _close(value: float, wanted: float) -> bool:
    return abs(value - wanted) <= CLOSE * max(abs(wanted), 1)


def score_figures(read: dict, keyed: dict, returned: list | None = None) -> dict:
    """The answer's figures, as read, counted against the key's; further figures by whether a tool returned them."""
    counts = dict.fromkeys(("right", "wrong", "superseded", "not_given", "hedged", "other_from_tools",
                            "other_unsourced"), 0)
    for name, figure in keyed["figures"].items():
        given = read["items"].get(name) or {}
        value = given.get("value")
        if value is None:
            counts["not_given"] += 1
        elif given.get("hedged"):
            counts["hedged"] += 1
        elif _close(value, figure["value"]):
            counts["right"] += 1
        else:
            counts["wrong"] += 1
            counts["superseded"] += any(_close(value, old) for old in keyed["superseded"])
    for other in read["other"]:
        if other.get("hedged"):
            counts["hedged"] += 1
        elif returned and not unsupported(f"{other['value']:g}", returned):
            counts["other_from_tools"] += 1
        else:
            counts["other_unsourced"] += 1
    return {**counts, "wrong_as_fact": counts["wrong"]}


def policy_cited(answer: str, keyed: dict) -> int:
    """How many of the key's handbook sections the answer names, by section number."""
    handbook = _handbook()
    named = 0
    for ident in keyed["citations"]:
        document = handbook[ident]
        section = document["citation"].rsplit("§", 1)[1].strip()
        named += bool(re.search(rf"§\s*{re.escape(section)}(?!\d)", answer))
    return named


def read_figures(client, answer: str, keyed: dict) -> dict:
    """The figures and verdict the answer gives, read by Claude, which is told what each figure is about but not
    its value."""
    value = {"type": "object", "additionalProperties": False, "required": ["value", "hedged"],
             "properties": {"value": {"type": ["number", "null"]}, "hedged": {"type": "boolean"}}}
    other = {"type": "object", "additionalProperties": False, "required": ["text", "value", "hedged"],
             "properties": {"text": {"type": "string"}, "value": {"type": "number"}, "hedged": {"type": "boolean"}}}
    schema = {"type": "object", "additionalProperties": False, "required": ["items", "other", "verdict"],
              "properties": {
                  "items": {"type": "object", "additionalProperties": False, "required": list(keyed["figures"]),
                            "properties": dict.fromkeys(keyed["figures"], value)},
                  "other": {"type": "array", "items": other},
                  "verdict": {"type": "string", "enum": [*keyed["choices"], NOT_GIVEN]}}}
    items = "\n".join(f"- {name}: {figure['about']}" for name, figure in keyed["figures"].items())
    response = client.messages.create(
        model=READER, max_tokens=4000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        system="You read an answer to an employee's question about their employer, Meridian Logistics, and report "
               "what it says, not what you think is right. For each item, give the number the answer states for "
               "it (dollars as plain numbers, percentages as numbers such as 12.5 for 12.5%, days or weeks as "
               "numbers), or null when it states none; hedged is true when the answer presents the number as a "
               "guess or what is typical elsewhere (\"typically\", \"many companies\", \"often\", \"usually\") "
               "rather than as Meridian's. Under other, list any further number the answer states about Meridian's "
               "policies or this employee that is none of the items, with its words; leave out dates, the "
               "employee's own figures quoted from the question, and section numbers. For verdict, choose the "
               "label that matches what the answer concludes, or \"not given\".",
        messages=[{"role": "user", "content": f"Items:\n{items}\n\nVerdict labels: "
                                              f"{', '.join(keyed['choices']) or '(none)'}\n\nThe answer:\n{answer}"}])
    return json.loads(next(block.text for block in response.content if block.type == "text"))


def wrong_layer(setup: str, folder: Path = HERE) -> str | None:
    """Why a derived setup cannot run with the layer now in derived/, or None."""
    wanted = LAYER_MODELS.get(setup)
    if wanted is None:
        return None
    model = derived.read(folder).get("model")
    if model != wanted:
        return f"{setup} needs the layer {wanted} derived, but derived/ holds {model or 'none'}: run measure.py " \
               f"restore {'claude' if wanted == decision_agent.MODEL else 'ollama'}"
    return None


def derive_layer(backend: str) -> int:
    """Derive the layer afresh with the backend, approve every entry that passed its check (a scripted approval,
    labelled so), and keep a copy in runs/measure/derived-BACKEND/."""
    shutil.rmtree(HERE / "derived", ignore_errors=True)
    code = derive.main(["run", str(DERIVED), "handbook", "--backend", backend, "--fresh", "--as-of", AS_OF])
    if code:
        return code
    layer = derived.read(HERE)
    for kind in ("decisions", "facts"):
        for name, entry in layer[kind].items():
            if entry["status"] == "checked":
                derived.set_status(HERE, kind, name, "approved", by=SCRIPTED)
    kept = RUNS / f"derived-{backend}"
    shutil.rmtree(kept, ignore_errors=True)
    shutil.copytree(HERE / "derived", kept)
    print(f"kept in {kept}", file=sys.stderr)
    return 0


def restore(backend: str) -> int:
    """Put a kept layer back in derived/."""
    kept = RUNS / f"derived-{backend}"
    if not kept.is_dir():
        print(f"error: no layer kept in {kept}: run measure.py derive {backend}", file=sys.stderr)
        return 2
    shutil.rmtree(HERE / "derived", ignore_errors=True)
    shutil.copytree(kept, HERE / "derived")
    return 0


def _bounds(table, name: str) -> set[float]:
    if name not in table.inputs:
        return set()
    column = table.inputs.index(name)
    return {float(found) for row in table.rules for found in re.findall(r"-?\d+(?:\.\d+)?", row.entries[column])}


def grid(hand, mine) -> list[dict]:
    """Every combination of: each bound in either table, one either side, and 0, for a number, none below 0 (every
    input here is a count of days or years); both booleans."""
    axes = []
    for index, name in enumerate(hand.inputs):
        if hand.types[index].endswith("boolean"):
            axes.append([True, False])
            continue
        bounds = _bounds(hand, name) | _bounds(mine, name)
        values = {0.0} | bounds | {bound + 1 for bound in bounds} | {bound - 1 for bound in bounds}
        axes.append(sorted(int(value) if value.is_integer() else value for value in values if value >= 0))
    return [dict(zip(hand.inputs, values)) for values in itertools.product(*axes)]


def _outputs(table, inputs: dict, names: list[str]):
    try:
        found = evaluate(table, inputs)["outputs"]
    except DmnError as error:
        return f"error: {error}"
    return {name: found.get(name) for name in names}


def _compare_table(name: str, folder: Path) -> list[dict]:
    """Where the derived table and the hand-written one give different outputs on the grid."""
    hand = load_table(HERE / "rules" / f"{name}.dmn")
    try:
        mine = load_table(folder / "rules" / f"{name}.dmn")
    except DmnError as error:
        return [{"inputs": None, "hand": None, "derived": f"error: {error}"}]
    names = [output for output in hand.outputs if output not in ("reason", "citation")]
    differences = []
    for inputs in grid(hand, mine):
        given = {key: value for key, value in inputs.items() if key in mine.inputs}
        expected, found = _outputs(hand, inputs, names), _outputs(mine, given, names)
        if expected != found:
            differences.append({"inputs": inputs, "hand": expected, "derived": found})
    return differences


def _hand_facts() -> dict:
    with open(HERE / "graph" / "nodes.csv", encoding="utf-8", newline="") as file:
        plan = next(row for row in csv.DictReader(file) if row["id"] == "k401_2026")
    return {**HAND_TIERS, "irs_limit": float(plan["irs_limit"]), "match_cap": float(plan["match_cap"])}


def compare(runs: Path = RUNS) -> dict:
    """Each kept layer against the second key: for each slot, whether it passed its check, how many tries it took,
    and whether it gives what the hand-written tables and figures give; with the entries checked but wrong."""
    slots = tomllib.loads(DERIVED.read_text(encoding="utf-8"))["derived"]
    names = [("decisions", name) for name in slots["decisions"]] + \
        [("facts", f"{node}.{prop}") for node, props in slots["facts"].items() for prop in props]
    facts, found = _hand_facts(), {}
    for folder in sorted(runs.glob("derived-*")):
        layer = json.loads((folder / "derived.json").read_text(encoding="utf-8"))
        rows = {}
        for kind, name in names:
            entry = layer[kind].get(name) or {}
            if kind == "decisions":
                differences = _compare_table(name, folder) if entry.get("dmn") else [{"derived": "no table"}]
            else:
                wanted = facts[name.split(".", 1)[1]]
                differences = [] if entry.get("value") == wanted else [{"hand": wanted, "derived": entry.get("value")}]
            rows[name] = {"kind": kind, "status": entry.get("status"), "attempts": entry.get("attempts"),
                          "check_passed": (entry.get("check") or {}).get("passed") is True,
                          "failures": (entry.get("check") or {}).get("failures", []),
                          "right": not differences, "differences": differences[:20]}
        summary = {"slots": len(rows), "right": sum(row["right"] for row in rows.values()),
                   "passed": sum(row["check_passed"] for row in rows.values()),
                   "passed_first_try": sum(row["check_passed"] and row["attempts"] == 1 for row in rows.values()),
                   "failed": sum(not row["check_passed"] for row in rows.values()),
                   "checked_but_wrong": sum(row["check_passed"] and not row["right"] for row in rows.values()),
                   "failed_but_right": sum(not row["check_passed"] and row["right"] for row in rows.values())}
        found[folder.name.removeprefix("derived-")] = {"model": layer.get("model"), "summary": summary,
                                                       "slots": rows}
    return found


def show_compare(found: dict) -> None:
    """The second key's results as a table."""
    for backend, result in found.items():
        print(f"{backend} ({result['model']}): {result['summary']}")
        for name, row in result["slots"].items():
            print(f"  {name:32} check {'passed' if row['check_passed'] else 'failed':6} tries {row['attempts']!s:4} "
                  f"{'right' if row['right'] else 'WRONG'}")


def run(backends: list[str], questions: list[str], setups: list[str]) -> int:
    """Ask every question in each setup, on each backend."""
    key = _key()
    RUNS.mkdir(parents=True, exist_ok=True)
    for setup in setups:
        refused = wrong_layer(setup)
        if refused:
            print(f"error: {refused}", file=sys.stderr)
            return 2
        environment, extra = SETUPS[setup]
        for backend in backends:
            for question in questions:
                out = RUNS / f"{question}-{setup}-{backend}.answer.json"
                print(out.name, file=sys.stderr, flush=True)
                code = decision_agent.main([str(environment), key[question]["question"], "--backend", backend,
                                            *extra, "--out", str(out)])
                if code:
                    print(f"  exit {code}", file=sys.stderr)
    return 0


def score() -> int:
    """Score every answer in runs/measure/ against the key, and write scores.json."""
    import anthropic  # pylint: disable=import-outside-toplevel
    client, key = anthropic.Anthropic(), _key()
    rows = []
    for path in sorted(RUNS.glob("*.answer.json")):
        question, setup, backend = path.name.removesuffix(".answer.json").split("-")
        result = json.loads(path.read_text(encoding="utf-8"))
        keyed = key[question]
        read = read_figures(client, result["answer"], keyed) if result["answer"] else {
            "items": {}, "other": [], "verdict": NOT_GIVEN}
        # What the tools returned, and the question, as clm-agent counts them
        returned = [call["error"] or call["result"] for call in result["calls"]] + [result["question"]]
        rows.append({"question": question, "setup": setup, "backend": backend, "model": result["model"],
                     **score_figures(read, keyed, returned), "figures": len(keyed["figures"]),
                     "verdict_right": None if not keyed["choices"] else read["verdict"] == keyed["verdict"],
                     "policy_cited": policy_cited(result["answer"], keyed), "policies": len(keyed["citations"]),
                     "unsupported": result["unsupported"], "calls": len(result["calls"]),
                     "seconds": result.get("seconds"), "stopped": result["stopped"], "read": read})
    (RUNS / "scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    show(rows)
    return 0


def show(rows: list[dict]) -> None:
    """The scores as a table."""
    print("question setup     backend  wrong-as-fact right   not-given hedged superseded unsourced verdict policy "
          "calls seconds")
    for r in rows:
        verdict = "" if r["verdict_right"] is None else "yes" if r["verdict_right"] else "no"
        print(f"{r['question']:8} {r['setup']:9} {r['backend']:8} {r['wrong_as_fact']:>13} "
              f"{r['right']:>3}/{r['figures']:<3} {r['not_given']:>9} {r['hedged']:>6} {r['superseded']:>10} "
              f"{r['other_unsourced']:>9} "
              f"{verdict:>7} {r['policy_cited']:>4}/{r['policies']} {r['calls']:>5} {r['seconds']!s:>7}"
              f"{'  stopped: ' + r['stopped'] if r['stopped'] else ''}")


def main(argv: list[str] | None = None) -> int:
    """run or score."""
    parser = argparse.ArgumentParser(description="Measure the five questions in three setups.")
    commands = parser.add_subparsers(dest="command", required=True)
    ran = commands.add_parser("run")
    ran.add_argument("--backends", default="claude,ollama")
    ran.add_argument("--questions", default="q1,q2,q3,q4,q5")
    ran.add_argument("--setups", default="none,rules,documents")
    commands.add_parser("score")
    commands.add_parser("compare")
    for command in ("derive", "restore"):
        commands.add_parser(command).add_argument("backend", choices=("claude", "ollama"))
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.backends.split(","), args.questions.split(","), args.setups.split(","))
    if args.command == "derive":
        return derive_layer(args.backend)
    if args.command == "restore":
        return restore(args.backend)
    if args.command == "compare":
        found = compare()
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "second_key.json").write_text(json.dumps(found, indent=2), encoding="utf-8")
        show_compare(found)
        return 0
    return score()


if __name__ == "__main__":
    sys.exit(main())
