"""Compare CLM with language models on the check cases: the same message, facts and options for every model.

    uv run --group demo python examples/crowdstrike/compare_models.py examples/crowdstrike/scenario.toml \\
        [--models clm,local,sonnet,haiku] [--limit N] [--workers 8]

Each language model reads the policy's route descriptions, the passenger's message, the situation's facts
and the journeys every rebooking route could book, fetched beforehand; it answers with a route and, where
it applies, a journey. It calls no tools. CLM decides the same cases through the server's own steps, without
actions. Scores are per route, as check.py gives them, plus an accuracy weighted by each route's share of
the day; time and cost are scaled to the day's 99,806 decisions. Results go to runs/, which git ignores.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import crowdstrike_tools
from check import score
from policy import ROUTES

HERE = Path(__file__).resolve().parent
DAY = 99806
# Each route's share of the day's right answers (seed 0, no seats held: Stage 3's count)
SHARES = {"Rebook": 0.469, "No_Action": 0.183, "Agent": 0.162, "Refund": 0.158, "Overnight": 0.022,
          "Rebook_Partner": 0.006}
CLAUDE = {"sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
LOCAL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
# US dollars per million tokens, from https://platform.claude.com/docs/en/about-claude/pricing on the date
# given; filled in on the day of the run, never from memory
PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_write": 2.5, "cache_read": 0.10},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},   # prompts to 100k
}
PRICES_DATE = "2026-10-22"
FORMAT = {"type": "json_schema", "schema": {
    "type": "object", "properties": {"route": {"type": "string", "enum": list(ROUTES)}, "option": {"type": "string"}},
    "required": ["route", "option"], "additionalProperties": False}}
SYSTEM = ("You decide what an airline does for a passenger whose flight was cancelled or badly delayed, under "
          "this policy. Choose exactly one of these answers:\n\n"
          + "\n\n".join(f"{name}: {text}" for name, text in ROUTES.items())
          + "\n\nYou are given the passenger's message, facts about their booking that the airline's systems "
          "worked out, and the journeys each rebooking answer could book. Reply with JSON only: "
          '{"route": "<one of the answers above>", "option": "<the id of the journey to book, for Rebook, '
          'Rebook_Partner or Overnight; otherwise an empty string>"}.')


def inputs(case: dict) -> dict:
    """What every model is given: the message, the situation's facts and each rebooking route's journeys."""
    request = {"text": case["text"], "context": case["context"]}
    return {"message": case["text"], "facts": crowdstrike_tools.situation(request)["facts"],
            "options": {route: crowdstrike_tools.alternatives({**request, "route": route})["options"]
                        for route in crowdstrike_tools.KINDS}}


def prompt(found: dict) -> tuple[str, str]:
    """(system, user): the system part is the same for every case, so it can be cached."""
    journeys = "\n".join(f"{route}:\n" + ("\n".join(f"- {option['id']}: {option['text']}" for option in options)
                                          or "- none")
                         for route, options in found["options"].items())
    facts = "\n".join(f"- {fact}" for fact in found["facts"])
    return SYSTEM, f"The passenger's message:\n{found['message']}\n\nKnown:\n{facts}\n\nJourneys:\n{journeys}"


def parse(text: str, found: dict) -> tuple[dict, bool]:
    """The reply as the server would give it, and whether it was usable. A reply that is not JSON, names no
    answer of the policy, or picks a journey its route does not offer, is unusable and counts as wrong."""
    unusable = {"route": "unusable", "uncertain": False, "option": None, "answers": {}}
    try:
        answer = json.loads(text)
        route, option = answer["route"], answer.get("option") or ""
    except (ValueError, TypeError, KeyError):
        return unusable, False
    if route not in ROUTES:
        return unusable, False
    if route in found["options"]:
        if option not in {offered["id"] for offered in found["options"][route]}:
            return unusable, False
        return {"route": route, "uncertain": False, "option": {"id": option}, "answers": {}}, True
    return {"route": route, "uncertain": False, "option": None, "answers": {}}, True


def claude_client():
    """The Anthropic client, from the usual credentials (ANTHROPIC_API_KEY or `ant auth login`)."""
    import anthropic  # pylint: disable=import-outside-toplevel
    return anthropic.Anthropic()


class ClaudeModel:
    """A Claude model through the Anthropic API: low effort, JSON output, the policy cached."""

    def __init__(self, model_id: str, client):
        self.model_id, self.client = model_id, client

    def decide(self, system: str, user: str) -> tuple[str, dict]:
        """(the reply's text, its token counts). A refusal comes back as an empty reply."""
        response = self.client.messages.create(
            model=self.model_id, max_tokens=1024,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low", "format": FORMAT})
        usage = response.usage
        counts = {"input": usage.input_tokens, "output": usage.output_tokens,
                  "cache_write": usage.cache_creation_input_tokens or 0,
                  "cache_read": usage.cache_read_input_tokens or 0}
        if response.stop_reason == "refusal":
            return "", counts
        return next((block.text for block in response.content if block.type == "text"), ""), counts


class LocalModel:
    """A local model on this machine, through MLX. Its tokens cost nothing; its time is what it costs."""

    def __init__(self, model_id: str):
        from generator import MLXGenerator  # pylint: disable=import-outside-toplevel
        self.generator = MLXGenerator(model_id, 200)

    def decide(self, system: str, user: str) -> tuple[str, dict]:
        """(the reply's text, its token counts)."""
        text = self.generator.generate(system, user)
        tokenizer = self.generator.tokenizer
        return text, {"input": len(tokenizer.encode(self.generator.build_prompt(system, user))),
                      "output": len(tokenizer.encode(text)), "cache_write": 0, "cache_read": 0}


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def summary(name: str, cases: list[dict], replies: list[dict], usage: list[dict], seconds: float,
            day: int = DAY) -> dict:
    """The scores per route, the accuracy weighted by the day's shares, and time and cost, per decision and
    for the day."""
    found = score(cases, replies)
    count = len(cases)
    price = PRICES.get(name)
    cost = (sum(sum(one[kind] * price[kind] for kind in price) for one in usage) / 1e6 / count) if price else 0.0
    weighted = sum(SHARES[route] * right / of for route, (right, of) in found["route"].items())
    tokens = {kind: round(sum(one[kind] for one in usage) / count) for kind in ("input", "output", "cache_read",
                                                                                 "cache_write")} if usage else {}
    return {"model": name, "cases": count, "route": found["route"], "option": found["option"],
            "right": sum(right for right, _ in found["route"].values()),
            "day_weighted": weighted, "unusable": sum(reply["route"] == "unusable" for reply in replies),
            "wrong_uncertain": found["wrong_uncertain"], "tokens_per_decision": tokens,
            "seconds_per_decision": seconds / count, "day_hours": seconds / count * day / 3600,
            "cost_per_decision": cost, "day_cost": cost * day}


def _ask(model, cases: list[dict], workers: int) -> tuple[list[dict], list[dict], float]:
    def one(case: dict) -> tuple[dict, dict]:
        found = inputs(case)
        text, counts = model.decide(*prompt(found))
        return parse(text, found)[0], counts

    start = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, cases))
    return [reply for reply, _ in results], [counts for _, counts in results], time.perf_counter() - start


def _clm(scenario, cases: list[dict]) -> tuple[list[dict], float]:
    import decision  # pylint: disable=import-outside-toplevel
    from clm_router import make_router  # pylint: disable=import-outside-toplevel
    from tools import load_tools  # pylint: disable=import-outside-toplevel
    scenario = replace(scenario, actions=())
    tools = load_tools(scenario)
    router, _ = make_router(scenario)
    start, replies = time.perf_counter(), []
    for first in range(0, len(cases), 32):
        replies += decision.decide_many([{"text": case["text"], "context": case["context"]}
                                         for case in cases[first:first + 32]], scenario, router, tools)
    return replies, time.perf_counter() - start


def _line(found: dict) -> str:
    return (f"{found['model']:<24}{found['right']:>4} of {found['cases']}  day-weighted {found['day_weighted']:.0%}  "
            f"unusable {found['unusable']:>3}  {found['seconds_per_decision'] * 1000:>7.0f} ms/decision  "
            f"day {found['day_hours']:>6.1f} h  ${found['day_cost']:,.0f}")


def main(argv: list[str] | None = None) -> int:  # pylint: disable=too-many-locals
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Compare CLM with language models on the CrowdStrike check cases.")
    parser.add_argument("scenario", help="path to the scenario file")
    parser.add_argument("--models", default="clm,local,sonnet,haiku")
    parser.add_argument("--cases", type=Path, default=HERE / "check.cases.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8, help="Claude calls at once")
    parser.add_argument("--local-model", default=LOCAL)
    parser.add_argument("--name", default=time.strftime("compare-%Y%m%d-%H%M%S"))
    parser.add_argument("--runs", type=Path, default=HERE / "runs", help="where the results are written")
    args = parser.parse_args(argv)
    from scenario import load_scenario  # pylint: disable=import-outside-toplevel
    scenario = load_scenario(args.scenario)
    crowdstrike_tools.configure(dict(scenario.tool_settings), HERE)
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line]
    cases = cases[:args.limit] if args.limit else cases
    results = []
    for name in args.models.split(","):
        if name == "clm":
            replies, seconds = _clm(scenario, cases)
            results.append(summary("clm", cases, replies, [], seconds))
        elif name == "local":
            replies, usage, seconds = _ask(LocalModel(args.local_model), cases, 1)
            results.append(summary(args.local_model, cases, replies, usage, seconds))
        else:
            try:
                client = claude_client()
                client.models.retrieve(CLAUDE[name])        # free, and fails without credentials
                model = ClaudeModel(CLAUDE[name], client)
            except Exception as error:  # pylint: disable=broad-exception-caught
                print(f"Skipping {name}: no Anthropic credentials ({type(error).__name__}: {error})")
                continue
            replies, usage, seconds = _ask(model, cases, args.workers)
            results.append(summary(CLAUDE[name], cases, replies, usage, seconds))
        print(_line(results[-1]), flush=True)
    args.runs.mkdir(exist_ok=True)
    (args.runs / f"{args.name}.json").write_text(json.dumps({"prices_date": PRICES_DATE, "results": results}, indent=2),
                                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[1]))
    sys.exit(main())
