"""Measure the three questions with and without the document tools, on Claude and on a local model.

    uv run python examples/noncompete/measure.py run [--backends claude,ollama] [--questions q1,q2,q3]
    uv run python examples/noncompete/measure.py score

run asks clm-agent each question with environment.toml and with environment_without_documents.toml, on each
backend, and writes each answer to runs/measure/, which git ignores. score reads them against answer_key.toml:

- verdicts right: Claude (claude-opus-5-5) reads which verdict, from a list for each question, the answer gives for
  each state; it judges nothing about the law. The comparison against the key is this script's.
- key citations cited: the key's citations the answer cites, by section (a subsection is not required) or by
  volume, reporter and page.
- citations not found: citations in the answer that no tool returned (citation_check.py).
- cases cited and found: Question 2's reporter citations that the tools returned.

Claude's credentials are the Anthropic SDK's usual ones; none are stored. Not legal advice.
"""
import argparse
import json
import sys
import tomllib
from pathlib import Path

from clm_decision import decision_agent
from clm_decision.citation_check import keys

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "measure"
ENVIRONMENTS = {"with": HERE / "environment.toml", "without": HERE / "environment_without_documents.toml"}
QUESTIONS = {
    "q1": "Does Meridian Logistics' standard non-compete agreement, standard_v3, hold in each state where we employ "
          "people? Give a verdict for each state, what must change, and the statute or case each comes from.",
    "q2": "In Texas, is a 2-year, nationwide non-compete for a sales manager reasonable? Our sales manager under "
          "standard_v3 serves Texas, Oklahoma and Louisiana. Will it hold, and how should we narrow it? Cite the "
          "statute and the cases.",
    "q3": "Employee e_1042 lives and works in Colorado, but their contract names Texas law. Which state's law "
          "governs their non-compete, does it hold, and on what statute?",
}
GRADER = "claude-opus-5-5"
NOT_GIVEN = "not given"
ENVIRONMENT_STATES = ("TX", "CA", "NY", "FL", "IL", "CO", "WA", "MN", "MA", "OK", "ND")


def _key() -> dict:
    return tomllib.loads((HERE / "answer_key.toml").read_text(encoding="utf-8"))


def expected(key: dict, question: str) -> dict[str, str]:
    """Each item the question asks a verdict for, and the key's verdict: a state's for q1, "TX" for q2, and the
    governing state and its verdict for q3."""
    if question == "q1":
        return {state: entry["holds"] for state, entry in key["q1"].items()}
    if question == "q2":
        return {key["q2"]["state"]: key["q2"]["holds"]}
    return {"governing": key["q3"]["governing"], "holds": key["q3"]["holds"]}


def citations(key: dict, question: str) -> list[str]:
    """The key's citations for a question."""
    if question == "q1":
        return [c for entry in key["q1"].values() for c in entry["citations"]]
    return key[question]["citations"]


def key_cited(answer: str, cited: list[str]) -> int:
    """How many of the key's citations the answer cites: a statute by its section, a case by its reporter citation."""
    written = {item[:2] for item in keys(answer)}
    return sum(1 for citation in dict.fromkeys(cited) if {item[:2] for item in keys(citation)} <= written)


def right(verdicts: dict[str, str], wanted: dict[str, str]) -> int:
    """How many items have the key's verdict."""
    return sum(1 for item, verdict in wanted.items() if verdicts.get(item) == verdict)


def read_verdicts(client, answer: str, wanted: dict[str, str], choices: list[str]) -> dict[str, str]:
    """Which of the choices the answer gives for each item: read by Claude, which is told nothing of the key."""
    schema = {"type": "object", "additionalProperties": False, "required": list(wanted),
              "properties": {item: {"type": "string",
                                    "enum": (sorted(ENVIRONMENT_STATES) if item == "governing" else choices)
                                    + [NOT_GIVEN]} for item in wanted}}
    response = client.messages.create(
        model=GRADER, max_tokens=2000, output_config={"effort": "low", "format": {"type": "json_schema",
                                                                                  "schema": schema}},
        system="You read an answer about non-compete agreements and report, for each item, which listed label "
               "matches what the answer says, or \"not given\" when the answer says nothing that matches one. "
               "Report what the answer says, not what you think is right. \"governing\" is the state whose law "
               "the answer says governs, as its two-letter code.",
        messages=[{"role": "user", "content": f"Labels: {', '.join(choices)}\n\nItems: {', '.join(wanted)}\n\n"
                                              f"The answer:\n{answer}"}])
    return json.loads(next(block.text for block in response.content if block.type == "text"))



def run(backends: list[str], questions: list[str]) -> int:
    """Ask every question with each environment file, on each backend."""
    RUNS.mkdir(parents=True, exist_ok=True)
    for documents, environment in ENVIRONMENTS.items():
        for backend in backends:
            for question in questions:
                out = RUNS / f"{question}-{documents}-{backend}.answer.json"
                print(f"{out.name}", file=sys.stderr, flush=True)
                code = decision_agent.main([str(environment), QUESTIONS[question], "--backend", backend,
                                            "--out", str(out)])
                if code:
                    print(f"  exit {code}", file=sys.stderr)
    return 0


def score() -> int:
    """Score every answer in runs/measure/ against the key, and write scores.json."""
    import anthropic  # pylint: disable=import-outside-toplevel
    client, key = anthropic.Anthropic(), _key()
    statutes = sorted({entry["holds"] for entry in key["q1"].values()})
    # Question 2 asks what Texas courts do with this covenant, so its verdicts are about the covenant as written
    choices = {"q1": statutes, "q2": [key["q2"]["holds"], "holds as written", "void"], "q3": statutes}
    rows = []
    for path in sorted(RUNS.glob("*.answer.json")):
        question, documents, backend = path.name.removesuffix(".answer.json").split("-")
        result = json.loads(path.read_text(encoding="utf-8"))
        wanted = expected(key, question)
        verdicts = read_verdicts(client, result["answer"], wanted, choices[question]) if result["answer"] else {}
        found = result["citations"]["found"]
        rows.append({"question": question, "documents": documents, "backend": backend, "model": result["model"],
                     "verdicts_right": right(verdicts, wanted), "verdicts": len(wanted), "read": verdicts,
                     "key_cited": key_cited(result["answer"], citations(key, question)),
                     "key_citations": len(dict.fromkeys(citations(key, question))),
                     "not_found": result["citations"]["not_found"],
                     "cases_found": sorted({c for c in found if any(i[0] == "reporter" for i in keys(c))})
                     if question == "q2" else None,
                     "calls": len(result["calls"]), "seconds": result.get("seconds"), "stopped": result["stopped"]})
    (RUNS / "scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    show(rows)
    return 0


def show(rows: list[dict]) -> None:
    """The scores as a table."""
    print("question documents backend   verdicts  key cites  not found  cases  calls  seconds")
    for r in rows:
        cases = "" if r["cases_found"] is None else len(r["cases_found"])
        print(f"{r['question']:8} {r['documents']:9} {r['backend']:8} {r['verdicts_right']:>3}/{r['verdicts']:<4} "
              f"{r['key_cited']:>4}/{r['key_citations']:<4} {len(r['not_found']):>9}  {cases!s:>5}  {r['calls']:>5}"
              f"  {r['seconds']!s:>7}{'  stopped: ' + r['stopped'] if r['stopped'] else ''}")


def main(argv: list[str] | None = None) -> int:
    """run or score."""
    parser = argparse.ArgumentParser(description="Measure the three questions with and without the documents.")
    commands = parser.add_subparsers(dest="command", required=True)
    ran = commands.add_parser("run")
    ran.add_argument("--backends", default="claude,ollama")
    ran.add_argument("--questions", default="q1,q2,q3")
    commands.add_parser("score")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.backends.split(","), args.questions.split(","))
    return score()


if __name__ == "__main__":
    sys.exit(main())
