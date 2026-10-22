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


def _options(outcome: dict) -> dict:
    """An options tool's outcome, checked: {"status", "facts", "options"} or {"status", "error"}."""
    if outcome["status"] == "failed":
        return outcome
    result = outcome["result"] if isinstance(outcome["result"], dict) else {}
    found = result.get("options")
    if not (isinstance(found, list) and all(isinstance(option, dict) and isinstance(option.get("id"), str)
                                            and isinstance(option.get("text"), str) for option in found)
            and len({option["id"] for option in found}) == len(found)):
        return {"status": "failed",
                "error": "an options tool must return {'options': [{'id': ..., 'text': ...}, ...]} with distinct ids"}
    checked = _facts({"status": "done", "result": {"facts": result.get("facts", [])}})
    return checked if checked["status"] == "failed" else {**checked, "options": found}


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


def heard(scenario: Scenario, informed: list[dict]) -> dict[str, list[str]]:
    """{question: the facts it hears}, from the informing tools' outcomes."""
    # Each question hears only the facts of the tools that inform it, since facts sway questions they have
    # nothing to do with
    found: dict[str, list] = {name: [] for name in scenario.questions}
    for outcome in informed:
        if outcome["status"] == "done" and outcome["facts"]:
            chosen = scenario.informs[outcome["tool"]].questions if outcome["tool"] in scenario.informs else None
            for question in scenario.questions if chosen is None else chosen:
                found[question] += outcome["facts"]
    return found


def _heard(scenario: Scenario, informed: list[dict]) -> dict[tuple, dict]:
    """{facts: the questions that hear exactly those facts}, from the informing tools' outcomes."""
    groups: dict[tuple, dict] = {}
    for name, facts in heard(scenario, informed).items():
        if facts:
            groups.setdefault(tuple(facts), {})[name] = scenario.questions[name]
    return groups


def gather(request: dict, scenario: Scenario, tools: dict[str, Callable], earlier: list[dict]) -> list[dict]:
    """Run the route's informing tools; their facts, and those of routing's tools in `earlier`, are added to
    `request`. A tool that already ran for routing is not run again."""
    informed = list(earlier)
    request["facts"] = facts_of(informed)
    route = scenario.routes[request["route"]]
    for name in route.inform:
        if name not in {outcome["tool"] for outcome in earlier}:
            # Each tool sees the facts found before it
            check = _options if route.options and name == route.options.tool else _facts
            informed.append({"tool": name, **check(_run(tools[name], dict(request), "inform"))})
            request["facts"] = facts_of(informed)
    return informed


def _asks(request: dict, scenario: Scenario, informed: list[dict]) -> list[tuple[str, dict]]:
    """The questions to ask again with facts: (the text with the facts, the questions that hear them)."""
    return [(routed_text(request["text"], list(facts)), asked)
            for facts, asked in _heard(scenario, informed).items()]


def _answer(request: dict, scenario: Scenario, router, asks: list[tuple[str, dict]]) -> None:
    """Ask the questions again with their facts; `request`'s answers and rules are updated."""
    answers = dict(request["answers"])
    for text, asked in asks:
        answers.update(router.ask(text, asked))
    if asks:
        request.update(answers=answers, rules=_rules(scenario, answers))


def _choose(request: dict, scenario: Scenario, router, informed: list[dict]) -> dict | None:
    """CLM's choice among the options the route's options tool fetched, or None if there are none. The
    acting tools get it as request["option"]."""
    spec = scenario.routes[request["route"]].options
    if spec is None:
        return None
    found = next((outcome for outcome in informed if outcome["tool"] == spec.tool), {})
    offered = found.get("options") if found.get("status") == "done" else None
    request["option"] = None
    if not offered:
        return None
    ranked = router.rank(routed_text(request["text"], request["facts"]),
                         {option["id"]: option["text"] for option in offered}, spec.instructions)
    chosen = next(option for option in offered if option["id"] == ranked["choice"])
    request["option"] = {"id": chosen["id"], "text": chosen["text"]}
    return {**request["option"], "confidence": ranked["confidence"], "probabilities": ranked["probabilities"],
            "uncertain": ranked["confidence"] < scenario.confidence_threshold}


def _option_problem(scenario: Scenario, route: str, option: dict | None) -> str | None:
    """Why a route with options cannot act: no option, or an uncertain one. None if it can."""
    if not scenario.routes[route].options:
        return None
    if option is None:
        return "no option"
    return "option uncertain" if option["uncertain"] else None


def _route(scenario: Scenario, made: dict, rerouted: dict | None) -> dict:
    """The route and the answers to the questions: CLM's decision on the message, with the route replaced by
    `rerouted`, the decision made with routing's facts, if there is one."""
    result = made
    if rerouted is not None:
        result = {key: value for key, value in made.items() if key not in ROUTE_KEYS}
        result.update(rerouted, before_facts={key: made[key] for key in ("route", "confidence", "probabilities")})
    # A route the owner chose is not second-guessed: the confidence is CLM's, in its own route
    result["uncertain"] = result.get("source") == "clm" and result["confidence"] < scenario.confidence_threshold
    return result


def _request(text: str, result: dict, scenario: Scenario, context: dict | None) -> dict:
    """What the tools are given, as far as the route and the answers."""
    request = {"text": text, "route": result["route"], "answers": result["answers"],
               "rules": _rules(scenario, result["answers"]), "facts": []}
    if context is not None:
        request["context"] = context
    return request


def _act(result: dict, request: dict, option: dict | None, scenario: Scenario,
         tools: dict[str, Callable]) -> list[dict]:
    """Run the acting tools the route and the answers call for, or say why each was skipped."""
    actions = []
    for action in scenario.actions:
        if result["route"] not in action.routes or (action.when and not action.when.applies(request["answers"])):
            continue
        skip = "route uncertain" if result["uncertain"] else _option_problem(scenario, result["route"], option)
        if skip:
            actions.append({"tool": action.tool, "status": "skipped", "reason": skip})
        else:
            actions.append({"tool": action.tool, **_run(tools[action.tool], dict(request), "act")})
    return actions


def _finish(made: dict, scenario: Scenario, router, tools: dict[str, Callable]) -> dict:
    """The reply for one request part-way through: `made` holds its result, request and informed outcomes,
    with the questions already asked again. Ranks the options and runs the acting tools."""
    result, request, informed = made["result"], made["request"], made["informed"]
    option = _choose(request, scenario, router, informed)
    reply = {**result, "answers": request["answers"], "rules": request["rules"], "informed": informed,
             "actions": _act(result, request, option, scenario, tools)}
    if scenario.routes[result["route"]].options:
        reply["option"] = option
    return reply


def decide(text: str, scenario: Scenario, router, tools: dict[str, Callable], context: dict | None = None) -> dict:
    """The scenario's decision on `text`, with what its tools found and did. `context` (such as a booking or
    user ID) is passed to the tools; CLM never reads it."""
    tracing.add("decisions")
    earlier = route_facts(text, scenario, tools, context)
    facts = facts_of(earlier)
    result = _route(scenario, router.decide(text, scenario),
                    router.route(routed_text(text, facts), scenario) if facts else None)
    request = _request(text, result, scenario, context)
    informed = gather(request, scenario, tools, earlier)
    _answer(request, scenario, router, _asks(request, scenario, informed))
    return _finish({"result": result, "request": request, "informed": informed}, scenario, router, tools)


def decide_many(requests: list[dict], scenario: Scenario, router, tools: dict[str, Callable]) -> list[dict]:
    """decide() for each of `requests` ({"text", "context"?}), with the same replies, made phase by phase so
    CLM encodes each phase's texts in one batched pass. The tools run in the order of the requests."""
    if not requests:
        return []
    tracing.add("decisions", len(requests))
    made = [{"text": request["text"], "context": request.get("context"),
             "earlier": route_facts(request["text"], scenario, tools, request.get("context"))} for request in requests]
    for one, decided in zip(made, router.decide_batch([one["text"] for one in made], scenario)):
        one["decided"] = decided
    with_facts = [one for one in made if facts_of(one["earlier"])]
    rerouted = router.route_batch([routed_text(one["text"], facts_of(one["earlier"])) for one in with_facts],
                                  scenario) if with_facts else []
    for one, routed in zip(with_facts, rerouted):
        one["rerouted"] = routed
    for one in made:
        one["result"] = _route(scenario, one["decided"], one.get("rerouted"))
        one["request"] = _request(one["text"], one["result"], scenario, one["context"])
        one["informed"] = gather(one["request"], scenario, tools, one["earlier"])
        one["asks"] = _asks(one["request"], scenario, one["informed"])
    # Texts asked the same questions are encoded together
    together: dict[tuple, tuple[list[str], dict]] = {}
    for text, questions in (pair for one in made for pair in one["asks"]):
        together.setdefault(tuple(questions), ([], questions))[0].append(text)
    for states, questions in together.values():
        router.prefetch(states, questions)
    for one in made:
        _answer(one["request"], scenario, router, one["asks"])
    return [_finish(one, scenario, router, tools) for one in made]


def check_call(text: str, call: dict, scenario: Scenario, router) -> dict:
    """CLM's opinion of a tool call the caller proposes for `text`."""
    p = router.check_tool(f"{text}\n\nProposed call: {json.dumps(call, sort_keys=True)}", scenario)
    return {"p": p, "approved": p >= scenario.tool_check_threshold}


def describe(scenario: Scenario) -> dict:
    """The decisions the scenario makes: its routes, questions and rules, and the facts the route considers."""
    described = {
        "routes": [{"name": route.name, "description": route.description,
                    **({"options": {"tool": route.options.tool, "instructions": route.options.instructions}}
                       if route.options else {})} for route in scenario.routes.values()],
        "questions": [{"name": name, "type": question["type"], "instructions": question["instructions"]}
                      for name, question in scenario.questions.items()],
        "rules": [{"when": rule.when, "tell": rule.tell} for rule in scenario.rules],
    }
    if scenario.route_inform:
        described["route_facts_from"] = list(scenario.route_inform)
    return described
