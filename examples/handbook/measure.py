"""Measure the five questions three ways, on Claude and on a local model: with no tools, with the rules only, and
with the rules and the handbook.

    uv run python examples/handbook/measure.py run [--backends claude,ollama] [--questions q1,...] [--setups none,...]
    uv run python examples/handbook/measure.py score

run asks clm-agent each question in each setup, on each backend, and writes each answer to runs/measure/, which git
ignores: `none` with --no-tools, `rules` with environment_without_documents.toml, `documents` with environment.toml.
score reads them against answer_key.toml:

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
import json
import re
import sys
import tomllib
from pathlib import Path

from clm_decision import decision_agent
from clm_decision.number_check import unsupported

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "measure"
SETUPS = {"none": (HERE / "environment.toml", ["--no-tools"]),
          "rules": (HERE / "environment_without_documents.toml", []),
          "documents": (HERE / "environment.toml", [])}
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


def run(backends: list[str], questions: list[str], setups: list[str]) -> int:
    """Ask every question in each setup, on each backend."""
    key = _key()
    RUNS.mkdir(parents=True, exist_ok=True)
    for setup in setups:
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
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.backends.split(","), args.questions.split(","), args.setups.split(","))
    return score()


if __name__ == "__main__":
    sys.exit(main())
