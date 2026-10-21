"""One decision: CLM picks the route, informing tools add facts, CLM answers the questions, actions run."""
from __future__ import annotations

import json
from typing import Callable

from scenario import Scenario


def _run(tool: Callable, request: dict) -> dict:
    """Call one tool with the request: {"status": "done", "result": ...} or {"status": "failed", "error": ...}."""
    # An owner's tool can fail in any way; the decision must still reach the caller
    try:
        result = tool(request)
        json.dumps(result)
    except Exception as error:  # pylint: disable=broad-exception-caught
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    return {"status": "done", "result": result}


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


def _inform(request: dict, scenario: Scenario, router, tools: dict[str, Callable]) -> list[dict]:
    """Run the route's informing tools; their facts are added to `request` and its answers and rules updated."""
    # Facts come after routing, so they never change the route; they inform the questions they bear on
    informed, informs = [], set()
    for name in scenario.routes[request["route"]].inform:
        outcome = _facts(_run(tools[name], dict(request)))
        informed.append({"tool": name, **outcome})
        if outcome["status"] == "done" and outcome["facts"]:
            request["facts"] = request["facts"] + outcome["facts"]
            chosen = scenario.informs[name].questions if name in scenario.informs else None
            informs.update(scenario.questions if chosen is None else chosen)
    asked = {name: question for name, question in scenario.questions.items() if name in informs}
    if asked:
        known = "\n".join(f"- {fact}" for fact in request["facts"])
        answers = {**request["answers"], **router.ask(f"{request['text']}\n\nKnown:\n{known}", asked)}
        request.update(answers=answers, rules=_rules(scenario, answers))
    return informed


def decide(text: str, scenario: Scenario, router, tools: dict[str, Callable]) -> dict:
    """The scenario's decision on `text`, with what its tools found and did."""
    result = router.decide(text, scenario)
    route = result["route"]
    result["uncertain"] = result["confidence"] < scenario.confidence_threshold
    request = {"text": text, "route": route, "answers": result["answers"],
               "rules": _rules(scenario, result["answers"]), "facts": []}

    informed = _inform(request, scenario, router, tools)

    actions = []
    for action in scenario.actions:
        if route not in action.routes or (action.when and not action.when.applies(request["answers"])):
            continue
        if result["uncertain"]:
            actions.append({"tool": action.tool, "status": "skipped", "reason": "route uncertain"})
        else:
            actions.append({"tool": action.tool, **_run(tools[action.tool], dict(request))})
    return {**result, "answers": request["answers"], "rules": request["rules"], "informed": informed,
            "actions": actions}


def check_call(text: str, call: dict, scenario: Scenario, router) -> dict:
    """CLM's opinion of a tool call the caller proposes for `text`."""
    p = router.check_tool(f"{text}\n\nProposed call: {json.dumps(call, sort_keys=True)}", scenario)
    return {"p": p, "approved": p >= scenario.tool_check_threshold}


def describe(scenario: Scenario) -> dict:
    """The decisions the scenario makes: its routes, questions and rules."""
    return {
        "routes": [{"name": route.name, "description": route.description} for route in scenario.routes.values()],
        "questions": [{"name": name, "type": question["type"], "instructions": question["instructions"]}
                      for name, question in scenario.questions.items()],
        "rules": [{"when": rule.when, "tell": rule.tell} for rule in scenario.rules],
    }
