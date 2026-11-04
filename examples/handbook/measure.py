"""Measure the five questions in seven setups, on Claude and on a local model: with no tools, with the rules only,
with the rules and the handbook, with the handbook only, with rules derived from the handbook by Claude or by the
local model, and with the rules and the handbook behind fewer tools.

    uv run python examples/handbook/measure.py derive claude|ollama
    uv run python examples/handbook/measure.py restore claude|ollama
    uv run python examples/handbook/measure.py recheck claude|ollama
    uv run python examples/handbook/measure.py run [--backends claude,ollama] [--questions q1,...] [--setups none,...]
    uv run python examples/handbook/measure.py compare
    uv run python examples/handbook/measure.py second claude|ollama claude|ollama MODEL
    uv run python examples/handbook/measure.py score

derive runs `clm-derive run --fresh` on environment_derived.toml with that backend, approves every entry that passed
its check, labelled as a scripted approval and not a person's, and keeps a copy of the layer in
runs/measure/derived-BACKEND/; restore puts that copy back; recheck checks that copy again with the check as it is
now, asking no model, and approves by script only what passes. compare evaluates each kept layer against the second
key, the hand-written rules/*.dmn and graph/nodes.csv, and writes runs/measure/second_key.json: above all, the
entries that passed the check and are wrong, which a person approving them would have had to catch. second checks a
copy of a kept layer again with a second reading by MODEL (clm-derive check, in runs/measure/second-FIRST-by-MODEL/),
asking no first model and leaving the kept layer as it is, and counts against the second key the wrong entries it
caught and the right ones it flagged, in runs/measure/second_reading.json.

run asks clm-agent each question in each setup, on each backend, and writes each answer to runs/measure/, which git
ignores: `none` with --no-tools, `rules` with environment_without_documents.toml, `documents` with environment.toml,
`documents_only` with environment_documents_only.toml, and `derived` and `derived_local` with
environment_derived.toml, which run only when the layer in derived/ is Claude's or gemma4's, and `fewer_tools` with a
copy of environment.toml in runs/measure/fewer_tools/ whose [serve] table offers only describe, the scenarios and the
search_ and document_ tools. score reads them against answer_key.toml:

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
import json
import re
import shutil
import sys
import tomllib
from datetime import date
from pathlib import Path

from clm_decision import decision_agent, derive, derive_check, derived
from clm_decision.dmn import DmnError, load_table
from clm_decision.environment import load_environment
from clm_decision.number_check import unsupported

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "measure"
DERIVED = HERE / "environment_derived.toml"
SETUPS = {"none": (HERE / "environment.toml", ["--no-tools"]),
          "rules": (HERE / "environment_without_documents.toml", []),
          "documents": (HERE / "environment.toml", []),
          "documents_only": (HERE / "environment_documents_only.toml", []),
          "derived": (DERIVED, []),
          "derived_local": (DERIVED, []),
          "fewer_tools": (None, [])}
# The fewer_tools setup's [serve] tools: describe, the scenarios and reading the handbook
FEWER_TOOLS = ["describe", "scenario_*", "search_*", "document_*"]
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
    _approve_checked(HERE)
    kept = RUNS / f"derived-{backend}"
    shutil.rmtree(kept, ignore_errors=True)
    shutil.copytree(HERE / "derived", kept)
    print(f"kept in {kept}", file=sys.stderr)
    return 0


def _approve_checked(folder: Path) -> None:
    layer = derived.read(folder)
    for kind in ("decisions", "facts"):
        for name, entry in layer[kind].items():
            if entry["status"] == "checked":
                derived.set_status(folder, kind, name, "approved", by=SCRIPTED)


def recheck_layer(backend: str, runs: Path = RUNS, folder: Path = HERE, env=None) -> int:
    """Check a kept layer again with the check as it is now, without asking any model: every entry is set back to
    proposed with no check, as a correction is, then checked, and the checked ones approved by script."""
    kept = runs / f"derived-{backend}"
    shutil.rmtree(folder / "derived", ignore_errors=True)
    shutil.copytree(kept, folder / "derived")
    layer = derived.read(folder)
    for kind in ("decisions", "facts"):
        for entry in layer[kind].values():
            entry.update(status="proposed", check=None, decided_by=None, decided_at=None, note=None)
    derived.write(folder, layer)
    day = date.fromisoformat(AS_OF)
    env = env or load_environment(folder / DERIVED.name, today=day)
    print(derive_check.recheck(folder, env, on=day), file=sys.stderr)
    _approve_checked(folder)
    shutil.rmtree(kept)
    shutil.copytree(folder / "derived", kept)
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


def _compare_table(name: str, folder: Path) -> list[dict]:
    """Where the derived table and the hand-written one give different outputs on the grid, compared as a second
    reading is (derive_check.table_differences)."""
    hand = load_table(HERE / "rules" / f"{name}.dmn")
    try:
        mine = load_table(folder / "rules" / f"{name}.dmn")
    except DmnError as error:
        return [{"inputs": None, "hand": None, "derived": f"error: {error}"}]
    names = [output for output in hand.outputs if output not in ("reason", "citation")]
    return [{"inputs": item["inputs"], "hand": item["first"], "derived": item["second"]}
            for item in derive_check.table_differences(hand, mine, names)]


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


SECOND = "a second reading by "


def _scratch(first: str, backend: str, model: str, runs: Path, here: Path) -> Path:
    """A copy of the derived variant, with the second reader named in its [derived] table, holding a copy of the
    kept layer with every entry back to proposed and no check, as a correction is; nothing tracked is touched."""
    scratch = runs / f"second-{first}-by-{re.sub(r'[^A-Za-z0-9._-]', '-', model)}"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    shutil.copy(here / "handbook_functions.py", scratch)
    shutil.copytree(here / "graph", scratch / "graph")
    text = (here / DERIVED.name).read_text(encoding="utf-8")
    (scratch / DERIVED.name).write_text(text.replace(
        "[derived]\n", f'[derived]\nsecond_reader = {{ backend = "{backend}", model = "{model}" }}\n', 1),
        encoding="utf-8")
    shutil.copytree(runs / f"derived-{first}", scratch / "derived")
    layer = derived.read(scratch)
    for kind in ("decisions", "facts"):
        for entry in layer[kind].values():
            entry.update(status="proposed", check=None, decided_by=None, decided_at=None, note=None)
    derived.write(scratch, layer)
    return scratch


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def second_read(first: str, backend: str, model: str, runs: Path = RUNS, here: Path = HERE) -> dict:
    """Check a copy of the kept layer derived-FIRST again with a second reading by the model, asking no first
    model, and count against the second key: the wrong entries it caught and the right ones it flagged (did not
    agree with), and how many pass both the mechanical check and the second reading. The kept layer is not
    changed."""
    scratch = _scratch(first, backend, model, runs, here)
    code = derive.main(["check", str(scratch / DERIVED.name), "--as-of", AS_OF])
    if code:
        raise SystemExit(code)
    key = compare(runs)[first]["slots"]
    layer, rows = derived.read(scratch), {}
    for name, keyed in key.items():
        check = layer[keyed["kind"]][name]["check"]
        reading = check.get("second_reading") or {}
        rows[name] = {"right": keyed["right"], "status": layer[keyed["kind"]][name]["status"],
                      "mechanical": not [line for line in check["failures"] if not line.startswith(SECOND)],
                      "reading": "not_made" if reading.get("error") else "agreed" if reading.get("agreed")
                      else "disagreed", "second_reading": reading, "failures": check["failures"]}
    flagged = [row for row in rows.values() if row["right"] and row["reading"] != "agreed"]
    both = [row for row in rows.values() if row["status"] == "checked"]
    summary = {"slots": len(rows), "wrong": sum(not row["right"] for row in rows.values()),
               "wrong_caught": sum(not row["right"] and row["reading"] != "agreed" for row in rows.values()),
               "right": sum(row["right"] for row in rows.values()), "right_flagged": len(flagged),
               "right_flagged_passing": sum(row["mechanical"] for row in flagged),
               **{reading: sum(row["reading"] == reading for row in rows.values())
                  for reading in ("agreed", "disagreed", "not_made")},
               "passes_both": len(both), "passes_both_but_wrong": sum(not row["right"] for row in both)}
    return {"first": first, "first_model": layer.get("model"), "reader": model, "summary": summary, "slots": rows}


def show_second(found: dict) -> None:
    """A second reading's results as a table."""
    print(f"{found['first_model']} read again by {found['reader']}: {found['summary']}")
    for name, row in found["slots"].items():
        check = 'passed' if row['mechanical'] else 'failed'
        print(f"  {name:32} {'right' if row['right'] else 'WRONG':5} check {check:6}"
              f" second {row['reading']:9} {'; '.join(line for line in row['failures'] if line.startswith(SECOND))}")


def show_compare(found: dict) -> None:
    """The second key's results as a table."""
    for backend, result in found.items():
        print(f"{backend} ({result['model']}): {result['summary']}")
        for name, row in result["slots"].items():
            print(f"  {name:32} check {'passed' if row['check_passed'] else 'failed':6} tries {row['attempts']!s:4} "
                  f"{'right' if row['right'] else 'WRONG'}")


def fewer_tools(runs: Path = RUNS, here: Path = HERE) -> Path:
    """A copy of environment.toml in runs/fewer_tools/, with the parts it reads beside it, whose [serve] table offers
    only FEWER_TOOLS; nothing tracked is touched."""
    folder = runs / "fewer_tools"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(here / "handbook_functions.py", folder)
    for part in ("graph", "rules"):
        shutil.copytree(here / part, folder / part)
    listed = ", ".join(f'"{name}"' for name in FEWER_TOOLS)
    text = (here / "environment.toml").read_text(encoding="utf-8")
    (folder / "environment.toml").write_text(f"{text}\n[serve]\ntools = [{listed}]\n", encoding="utf-8")
    return folder / "environment.toml"


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
        environment = environment or fewer_tools()
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


def _second(first: str, backend: str, model: str) -> int:
    found = second_read(first, backend, model)
    out = RUNS / "second_reading.json"
    kept = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    kept[f"{first}-by-{model}"] = found
    out.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    show_second(found)
    return 0


def main(argv: list[str] | None = None) -> int:  # pylint: disable=too-many-return-statements
    """run or score."""
    parser = argparse.ArgumentParser(description="Measure the five questions in several setups.")
    commands = parser.add_subparsers(dest="command", required=True)
    ran = commands.add_parser("run")
    ran.add_argument("--backends", default="claude,ollama")
    ran.add_argument("--questions", default="q1,q2,q3,q4,q5")
    ran.add_argument("--setups", default="none,rules,documents")
    commands.add_parser("score")
    commands.add_parser("compare")
    for command in ("derive", "restore", "recheck"):
        commands.add_parser(command).add_argument("backend", choices=("claude", "ollama"))
    again = commands.add_parser("second")
    again.add_argument("first", choices=("claude", "ollama"), help="the kept layer to read again")
    again.add_argument("backend", choices=("claude", "ollama"), help="the second reader's backend")
    again.add_argument("model", help="the second reader's model")
    args = parser.parse_args(argv)
    if args.command == "second":
        return _second(args.first, args.backend, args.model)
    if args.command == "run":
        return run(args.backends.split(","), args.questions.split(","), args.setups.split(","))
    if args.command == "derive":
        return derive_layer(args.backend)
    if args.command == "restore":
        return restore(args.backend)
    if args.command == "recheck":
        return recheck_layer(args.backend)
    if args.command == "compare":
        found = compare()
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "second_key.json").write_text(json.dumps(found, indent=2), encoding="utf-8")
        show_compare(found)
        return 0
    return score()


if __name__ == "__main__":
    sys.exit(main())
