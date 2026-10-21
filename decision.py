"""One decision: CLM picks the route, informing tools add facts, CLM answers the questions, actions run."""
from __future__ import annotations

import json
import time
from typing import Callable

import tracing
from scenario import Scenario

# A decision's fields that say which route and how sure; routing again with facts replaces them
ROUTE_KEYS = ("route", "confidence", "probabilities", "source", "clm_route", "similarity")


def _run(tool: Callable, request: dict, kind: str) -> dict:
    """Call one tool with the request: {"status": "done", "result": ...} or {"status": "failed", "error": ...}.
    `kind` ("inform" or "act") is for the trace."""
    start = time.perf_counter()
    # An owner's tool can fail in any way; the decision must still reach the caller
    try:
        result = tool(request)
        json.dumps(result)
        outcome = {"status": "done", "result": result}
    except Exception as error:  # pylint: disable=broad-exception-caught
        outcome = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    tracing.note_tool(tool.__name__, kind, (time.perf_counter() - start) * 1000, outcome["status"])
    return outcome


def _facts(outcome: dict) -> dict:
    """An informing tool's outcome, with its facts checked: {"status", "facts"} or {"status", "error"}."""
    if outcome["status"] == "failed":
        return outcome
    facts = outcome["result"].get("facts") if isinstance(outcome["result"], dict) else None
    if not isinstance(facts, list) or not all(isinstance(fact, str) for fact in facts):
        return {"status": "failed", "error": "an informing tool must return {'facts': [sentence, ...]}"}
    return {"status": "done", "facts": facts}


def _rules(scenario: Scenario, answers: dict) -> list[dict]:
    return [{"when": rule.when, "tell": rule.tell} for rule in scenario.matching_rules(answers)]


def routed_text(text: str, facts: list[str]) -> str:
    """The text CLM reads: the request, then the facts known about it, if any."""
    if not facts:
        return text
    return f"{text}\n\nKnown:\n" + "\n".join(f"- {fact}" for fact in facts)


def facts_of(informed: list[dict]) -> list[str]:
    """The facts the informing tools in `informed` found, in order."""
    return [fact for outcome in informed if outcome["status"] == "done" for fact in outcome["facts"]]


def route_facts(text: str, scenario: Scenario, tools: dict[str, Callable], context: dict | None = None) -> list[dict]:
    """Run routing's informing tools ([routing] inform), before there is a route; -> their outcomes."""
    request: dict = {"text": text, "route": None, "answers": {}, "rules": [], "facts": []}
    if context is not None:
        request["context"] = context
    return [{"tool": name, **_facts(_run(tools[name], dict(request), "inform"))} for name in scenario.route_inform]


def _heard(scenario: Scenario, informed: list[dict]) -> dict[tuple, dict]:
    """{facts: the questions that hear exactly those facts}, from the informing tools' outcomes."""
    # Each question hears only the facts of the tools that inform it, since facts sway questions they have
    # nothing to do with
    heard: dict[str, list] = {name: [] for name in scenario.questions}
    for outcome in informed:
        if outcome["status"] == "done" and outcome["facts"]:
            chosen = scenario.informs[outcome["tool"]].questions if outcome["tool"] in scenario.informs else None
            for question in scenario.questions if chosen is None else chosen:
                heard[question] += outcome["facts"]
    groups: dict[tuple, dict] = {}
    for name, facts in heard.items():
        if facts:
            groups.setdefault(tuple(facts), {})[name] = scenario.questions[name]
    return groups


def _inform(request: dict, scenario: Scenario, router, tools: dict[str, Callable], earlier: list[dict]) -> list[dict]:
    """Run the route's informing tools; their facts, and those of routing's tools in `earlier`, are added to
    `request` and given to the questions they inform. A tool that already ran for routing is not run again."""
    informed = list(earlier)
    request["facts"] = facts_of(informed)
    for name in scenario.routes[request["route"]].inform:
        if name not in {outcome["tool"] for outcome in earlier}:
            # Each tool sees the facts found before it
            informed.append({"tool": name, **_facts(_run(tools[name], dict(request), "inform"))})
            request["facts"] = facts_of(informed)
    groups = _heard(scenario, informed)
    answers = dict(request["answers"])
    for facts, asked in groups.items():
        answers.update(router.ask(routed_text(request["text"], list(facts)), asked))
    if groups:
        request.update(answers=answers, rules=_rules(scenario, answers))
    return informed


def decide(text: str, scenario: Scenario, router, tools: dict[str, Callable], context: dict | None = None) -> dict:
    """The scenario's decision on `text`, with what its tools found and did. `context` (such as a booking or
    user ID) is passed to the tools; CLM never reads it."""
    earlier = route_facts(text, scenario, tools, context)
    result = router.decide(text, scenario)
    if facts_of(earlier):
        before = result
        result = {key: value for key, value in before.items() if key not in ROUTE_KEYS}
        result.update(router.route(routed_text(text, facts_of(earlier)), scenario),
                      before_facts={key: before[key] for key in ("route", "confidence", "probabilities")})
    route = result["route"]
    # A route the owner chose is not second-guessed: the confidence is CLM's, in its own route
    result["uncertain"] = result.get("source") == "clm" and result["confidence"] < scenario.confidence_threshold
    request = {"text": text, "route": route, "answers": result["answers"],
               "rules": _rules(scenario, result["answers"]), "facts": []}
    if context is not None:
        request["context"] = context

    informed = _inform(request, scenario, router, tools, earlier)

    actions = []
    for action in scenario.actions:
        if route not in action.routes or (action.when and not action.when.applies(request["answers"])):
            continue
        if result["uncertain"]:
            actions.append({"tool": action.tool, "status": "skipped", "reason": "route uncertain"})
        else:
            actions.append({"tool": action.tool, **_run(tools[action.tool], dict(request), "act")})
    return {**result, "answers": request["answers"], "rules": request["rules"], "informed": informed,
            "actions": actions}


def check_call(text: str, call: dict, scenario: Scenario, router) -> dict:
    """CLM's opinion of a tool call the caller proposes for `text`."""
    p = router.check_tool(f"{text}\n\nProposed call: {json.dumps(call, sort_keys=True)}", scenario)
    return {"p": p, "approved": p >= scenario.tool_check_threshold}


def describe(scenario: Scenario) -> dict:
    """The decisions the scenario makes: its routes, questions and rules, and the facts the route considers."""
    described = {
        "routes": [{"name": route.name, "description": route.description} for route in scenario.routes.values()],
        "questions": [{"name": name, "type": question["type"], "instructions": question["instructions"]}
                      for name, question in scenario.questions.items()],
        "rules": [{"when": rule.when, "tell": rule.tell} for rule in scenario.rules],
    }
    if scenario.route_inform:
        described["route_facts_from"] = list(scenario.route_inform)
    return described
