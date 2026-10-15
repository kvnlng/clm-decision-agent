"""Build the LangGraph agent from a scenario: CLM decides, the generator writes."""
from __future__ import annotations

import json
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from generator import GeneratorError
from scenario import Scenario
from tools import resolve_tools

CLARIFY_PROMPT = (
    "The user's request could not be matched to a category with confidence. "
    "The two most likely categories are:\n{options}\n"
    "Ask the user one short question that would tell these apart. Do not answer the request yet."
)
DECLINED = ("This tool call was not permitted, so you have no data from it. Tell the user you could not "
            "look this up. Do not state or guess any details the tool would have returned.")
USER_DECLINED = "The user chose not to run this yet. Ask what they would like to change."
EXTRACT_PROMPT = (
    "You extract details from a conversation. Return only a JSON object with these keys:\n{fields}\n"
    "Use the user's own words, shortened. Use null for anything the user has not stated. "
    "A reply such as \"not sure\" or \"I don't know\" is not a value: use null. Do not guess."
)
ASK_PROMPT = (
    "\n\nYou are collecting the details needed before this request can be submitted.\n"
    "Already have:\n{have}\nStill needed:\n{need}\n"
    "Ask the user for the details still needed, in one short message. Ask for nothing else, "
    "and do not submit anything yet."
)
ACT_PROMPT = (
    "\n\nAll the details needed are collected:\n{have}\n"
    "Use your tools to look up anything you need, then submit the request with the submitting tool. "
    "Write it from these details and any guidance above. Do not ask the user for more details."
)
DONE_PROMPT = (
    "\n\nThe request has been submitted. Tell the user its reference and what happens next, "
    "in two or three sentences."
)
FOLLOW_UP_PROMPT = (
    "\n\nThe user's last request has already been submitted, and their new message does not describe "
    "another one. Reply to it in a sentence or two. Do not ask for details and do not submit anything. "
    "If they seem to want something else, ask what they need."
)
STOPPED_PROMPT = (
    "\n\nThe request was not submitted: {reason}. The details collected so far are kept:\n{have}\n"
    "Tell the user this in one or two sentences and ask what they would like to do. Do not claim that "
    "anything was submitted."
)
CATALOGUE_QID = "catalogue"
NON_ANSWERS = frozenset({"", "null", "none", "unknown", "n/a", "not stated", "not specified"})
# A small model often copies the user's whole non-answer ("not sure yet, my manager asked") into the field
NON_ANSWER_STARTS = ("not sure", "unsure", "i'm not sure", "i am not sure", "don't know", "i don't know",
                     "do not know", "no idea", "not yet decided", "to be decided")


class AgentState(TypedDict, total=False):
    """What the graph carries through one turn."""

    messages: Annotated[list, add_messages]
    decision: dict
    tool_checks: list
    tool_rounds: int
    draft: dict
    draft_route: str | None
    draft_start: int
    follow_up: bool
    intake: dict | None
    submitted: bool


def render_state(messages: list, history_turns: int) -> str:
    """The conversation as prose for CLM: the last `history_turns` user turns and the replies between."""
    lines, users = [], 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
            users += 1
            if users == history_turns:
                break
        elif isinstance(message, AIMessage) and message.content:
            lines.append(f"Assistant: {message.content}")
    return "\n".join(reversed(lines))


def plain_history(messages: list) -> list:
    """The conversation without tool calls and tool results, for prompts that offer no tools."""
    return [AIMessage(content=m.content) if isinstance(m, AIMessage) else m
            for m in messages
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.content)]


def signals_line(answers: dict) -> str:
    """CLM's answers to the extra questions, as a line appended to a route's system prompt."""
    parts = []
    for name, answer in answers.items():
        if answer["type"] == "noul":
            parts.append(f"{name} {answer['noul']:.0%}")
        elif answer["type"] == "score":
            parts.append(f"{name} {answer['score']:.1f}/{len(answer['probabilities']) - 1}")
        else:
            parts.append(f"{name} {answer['choice']}")
    return "\n\nSignals: " + ", ".join(parts) if parts else ""


def describe_calls(calls: list[dict]) -> str:
    """Tool calls as text, one `- name({args})` line each."""
    return "\n".join(f"- {call['name']}({json.dumps(call['args'], sort_keys=True)})" for call in calls)


def guidance_block(scenario: Scenario, answers: dict) -> str:
    """What the scenario's rules tell the generator to do, given CLM's answers."""
    tells = [rule.tell for rule in scenario.matching_rules(answers)]
    return "\n\nGuidance:\n" + "\n".join(f"- {tell}" for tell in tells) if tells else ""


def facts_block(entry: dict) -> str:
    """What is known about a picked catalogue entry, as a block for CLM's state and the model's prompt."""
    facts = entry.get("facts") or []
    return f"\n\nKnown about {entry['name']}:\n" + "\n".join(f"- {fact}" for fact in facts) if facts else ""


def parse_fields(text: str, fields: dict) -> dict | None:
    """The field values stated in the generator's extraction output; None if it holds no JSON object."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    found = {}
    for name in fields:
        value = data.get(name)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            value = str(value).strip()
            lowered = value.lower()
            if lowered not in NON_ANSWERS and not lowered.startswith(NON_ANSWER_STARTS):
                found[name] = value
    return found


# The nodes are closures over the scenario, router and generator, so they live in one function.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
def build_graph(scenario: Scenario, router, generator, registry: dict | None = None,
                review: bool = False, record=None, catalogue: list[dict] | None = None):
    """`review=True` adds nodes that pause for a human verdict and pass each verdict to `record`."""
    route_tools = resolve_tools(scenario, registry)
    all_tools = {tool.name: tool for tools in route_tools.values() for tool in tools}
    record = record or (lambda entry: None)

    entries = {entry["name"]: entry for entry in catalogue or []}

    def consult(state: AgentState, route) -> dict:
        """Once a turn on a catalogue route: CLM picks an entry; a confident pick adds its facts and re-asks."""
        decision, settings = state["decision"], scenario.catalogue
        if not entries or route.name not in settings.routes or CATALOGUE_QID in decision:
            return decision
        # The routing text, without facts: the pick is about what the user said
        text = decision["state_text"]
        question = {"type": "choice", "instructions": settings.instructions,
                    "criteria": {name: entry["description"] for name, entry in entries.items()}}
        answer = router.ask(text, {CATALOGUE_QID: question})[CATALOGUE_QID]
        picked = answer["confidence"] >= settings.confidence_threshold
        decision = {**decision, CATALOGUE_QID: {"choice": answer["choice"], "confidence": answer["confidence"],
                                                "probabilities": answer["probabilities"], "picked": picked}}
        informed = {name: scenario.questions[name] for name in settings.questions}
        if picked and informed:
            # Only the questions the facts bear on: facts also sway unrelated ones, such as urgency
            answers = {**decision["answers"], **router.ask(text + facts_block(entries[answer["choice"]]), informed)}
            decision.update(answers=answers, rules=[rule.when for rule in scenario.matching_rules(answers)])
        return decision

    def known(decision: dict) -> str:
        """The picked entry's facts for the model's prompt; empty without a confident pick."""
        pick = decision.get(CATALOGUE_QID)
        return facts_block(entries[pick["choice"]]) if pick and pick["picked"] else ""

    def needs_confirm(check: dict) -> bool:
        return any(call["name"] in scenario.confirm_tools for call in check["calls"])
    graph = StateGraph(AgentState)

    def assess(state: AgentState) -> dict:
        text = render_state(state["messages"], scenario.history_turns)
        decision = {**router.decide(text, scenario), "state_text": text}
        decision["rules"] = [rule.when for rule in scenario.matching_rules(decision["answers"])]
        open_route = state.get("draft_route")
        if open_route in scenario.routes and scenario.routes[open_route].fields:
            # An intake in progress keeps its route: a reply such as "about three months" cannot
            # be classified on its own. CLM's choice is kept for display only.
            decision.update(route=open_route, source="intake",
                            clm_route=decision.get("clm_route", decision["route"]))
        # The chat loop passes the whole history each turn, so per-turn state starts over here
        return {"decision": decision, "tool_checks": [], "tool_rounds": 0, "submitted": False, "intake": None}

    def route_node(route):
        tools = route_tools[route.name]

        def node(state: AgentState) -> dict:
            decision = consult(state, route)
            answers = decision["answers"]
            system = route.system_prompt + signals_line(answers) + guidance_block(scenario, answers) + known(decision)
            offered = tools if tools and state["tool_rounds"] < scenario.max_tool_rounds else None
            # Earlier tool traffic teaches a model the tool-call format, so routes without tools do not see it
            history = state["messages"] if tools else plain_history(state["messages"])
            return {"messages": [generator.generate(system, history, tools=offered)], "decision": decision}
        return node

    def stopped(state: AgentState) -> str | None:
        """Why a complete request cannot be submitted in this turn, if it cannot."""
        last, checks = state["messages"][-1], state["tool_checks"]
        if isinstance(last, ToolMessage) and checks and needs_confirm(checks[-1]):
            # settle() would have closed the intake had the confirm tool succeeded
            return "the user chose not to go ahead" if not checks[-1]["approved"] else \
                f"the submitting tool failed ({last.content})"
        if state["tool_rounds"] >= scenario.max_tool_rounds:
            return "too many tool calls were needed in one turn"
        return None

    def intake_node(route):
        tools = route_tools[route.name]

        def node(state: AgentState) -> dict:  # pylint: disable=too-many-locals
            decision = consult(state, route)
            messages, answers = state["messages"], decision["answers"]
            base = route.system_prompt + signals_line(answers) + guidance_block(scenario, answers) + known(decision)
            if state.get("submitted"):
                try:
                    reply = generator.generate(base + DONE_PROMPT, messages)
                except GeneratorError:
                    # The submission has happened; losing this turn would invite the user to submit it again
                    reply = AIMessage(
                        content=f"Your request was submitted. The service answered: {messages[-1].content}")
                # The next request starts after this reply, so it is not filled from this conversation
                return {"messages": [reply], "draft_start": len(messages) + 1, "follow_up": True, "decision": decision}

            opened = state.get("draft_route") == route.name
            draft = dict(state.get("draft") or {}) if opened else {}
            # Everything said since the last submission (or `new`) belongs to this request
            start = state.get("draft_start", 0)
            if isinstance(messages[-1], HumanMessage):      # the start of a turn, not a return from a tool
                listing = "\n".join(f'- "{name}": {ask}' for name, ask in route.fields.items())
                extraction = generator.generate(EXTRACT_PROMPT.format(fields=listing), plain_history(messages[start:]))
                found = parse_fields(str(extraction.content), route.fields) or {}
                if scenario.catalogue and draft.get(scenario.catalogue.field) in entries:
                    # A catalogue name CLM chose earlier is not replaced by the user's own words for it
                    found.pop(scenario.catalogue.field, None)
                draft.update(found)
            if state.get("follow_up") and not draft:
                # CLM always picks a route, so "thanks" after a submission lands on one. A message that
                # holds no detail of a new request gets a reply; it does not start the questions again.
                # Checked before the pick: CLM still reads the submitted request and may pick from it.
                return {"messages": [generator.generate(base + FOLLOW_UP_PROMPT, plain_history(messages))],
                        "decision": decision}
            pick = decision.get(CATALOGUE_QID)
            if pick and pick["picked"]:
                draft[scenario.catalogue.field] = pick["choice"]

            # A request that was already submitted stays out of view: a small model will otherwise
            # repeat its "submitted" reply instead of working on the new one
            current = messages[start:]
            missing = [name for name in route.fields if name not in draft]
            have = "\n".join(f"- {route.fields[name]}: {draft[name]}" for name in route.fields if name in draft)
            if missing:
                need = "\n".join(f"- {route.fields[name]}" for name in missing[:2])
                prompt = base + ASK_PROMPT.format(have=have or "- nothing yet", need=need)
                reply = generator.generate(prompt, plain_history(current))
            elif stopped(state):
                reply = generator.generate(base + STOPPED_PROMPT.format(reason=stopped(state), have=have), current)
            else:
                reply = generator.generate(base + ACT_PROMPT.format(have=have), current, tools=tools)
            progress = {"filled": len(route.fields) - len(missing), "total": len(route.fields), "missing": missing}
            return {"messages": [reply], "draft": draft, "draft_route": route.name, "draft_start": start,
                    "follow_up": False,
                    "intake": progress, "decision": decision}
        return node

    def clarify(state: AgentState) -> dict:
        probabilities = state["decision"]["probabilities"]
        top = sorted(probabilities, key=probabilities.get, reverse=True)[:2]
        options = "\n".join(f"- {name}: {scenario.routes[name].description}" for name in top)
        return {"messages": [generator.generate(CLARIFY_PROMPT.format(options=options),
                                                plain_history(state["messages"]))]}

    def verify_tool(state: AgentState) -> dict:
        """CLM checks the generator's proposed tool calls before anything runs."""
        route = scenario.routes[state["decision"]["route"]]
        calls = [{"name": call["name"], "args": call["args"]} for call in state["messages"][-1].tool_calls]
        if any(call["name"] not in route.tools for call in calls):
            check = {"calls": calls, "p": 0.0, "approved": False, "source": "unknown_tool"}
        else:
            text = (render_state(state["messages"], scenario.history_turns)
                    + "\n\nProposed tool call:\n" + describe_calls(calls))
            p = router.check_tool(text, scenario)
            check = {"calls": calls, "p": p, "approved": p >= scenario.tool_check_threshold,
                     "source": "clm", "state_text": text}
        return {"tool_checks": state["tool_checks"] + [check], "tool_rounds": state["tool_rounds"] + 1}

    def decline_tool(state: AgentState) -> dict:
        check = state["tool_checks"][-1]
        by_person = check["source"] in ("user", "human") and needs_confirm(check)
        # Every tool call needs an answering ToolMessage or the next prompt is malformed
        return {"messages": [ToolMessage(content=USER_DECLINED if by_person else DECLINED,
                                         tool_call_id=call["id"], name=call["name"])
                             for call in state["messages"][-1].tool_calls]}

    def settle(state: AgentState) -> dict:
        """After tools run: a confirm tool that succeeded completes the intake."""
        calls = state["tool_checks"][-1]["calls"]
        results = [m for m in state["messages"] if isinstance(m, ToolMessage)][-len(calls):]
        if any(m.name in scenario.confirm_tools and m.status != "error" for m in results):
            return {"draft": {}, "draft_route": None, "intake": None, "submitted": True}
        return {}

    # Review nodes. On resume a node runs again from its first line and interrupt() returns the
    # human's answer, so side effects (the record) come after the interrupt() call.

    def review_route(state: AgentState) -> dict:
        decision = state["decision"]
        answer = interrupt({"kind": "route", "decision": decision, "routes": list(scenario.routes)})
        # When a past review supplied the route, CLM's own choice is in clm_route
        clm_route = decision.get("clm_route", decision["route"])
        record({"kind": "route", "state_text": decision["state_text"], "human": answer["route"],
                "clm": {"route": clm_route, "confidence": decision["confidence"],
                        "probabilities": decision["probabilities"]}})
        return {"decision": {**decision, "route": answer["route"], "source": "human", "clm_route": clm_route}}

    def review_tool(state: AgentState) -> dict:
        check = state["tool_checks"][-1]
        approved = bool(interrupt({"kind": "tool", "check": check})["approved"])
        record({"kind": "tool", "state_text": check["state_text"], "calls": check["calls"],
                "clm": {"p": check["p"], "approved": check["approved"]}, "human": approved})
        reviewed = {**check, "approved": approved, "source": "human", "clm_approved": check["approved"]}
        return {"tool_checks": state["tool_checks"][:-1] + [reviewed]}

    def confirm_tool(state: AgentState) -> dict:
        """Run mode: a tool that changes something waits for the user's yes. CLM's check is advice."""
        check = state["tool_checks"][-1]
        approved = bool(interrupt({"kind": "confirm", "check": check})["approved"])
        confirmed = {**check, "approved": approved, "source": "user", "clm_approved": check["approved"]}
        return {"tool_checks": state["tool_checks"][:-1] + [confirmed]}

    def review_reply(state: AgentState) -> dict:
        decision = state["decision"]
        reply = str(state["messages"][-1].content)
        # Off-route tool calls are declined without a review pause, so the reviewer sees them here
        off_route = [check for check in state["tool_checks"] if check["source"] == "unknown_tool"]
        answer = interrupt({"kind": "reply", "route": decision["route"], "reply": reply,
                            "tool_checks": off_route, "catalogue": decision.get("catalogue")})
        record({"kind": "reply", "state_text": decision["state_text"], "route": decision["route"],
                "reply": reply, "off_route_calls": [call for check in off_route for call in check["calls"]],
                "human": {"acceptable": bool(answer["acceptable"]), "note": answer.get("note", "")}})
        return {}

    reply_done = "review_reply" if review else END

    def after_assess(state: AgentState) -> str:
        decision = state["decision"]
        # A route chosen by a person (now or in a past review) is not second-guessed
        if decision["source"] != "clm" or decision["confidence"] >= scenario.confidence_threshold:
            return decision["route"]
        return "clarify"

    def after_assess_in_review(state: AgentState) -> str:
        # An open intake already has its route; the reviewer is asked only when one starts
        return after_assess(state) if state["decision"]["source"] == "intake" else "review_route"

    def after_route(state: AgentState) -> str:
        return "verify_tool" if state["messages"][-1].tool_calls else reply_done

    def after_verify(state: AgentState) -> str:
        check = state["tool_checks"][-1]
        if check["source"] == "unknown_tool":
            return "decline_tool"
        if review:
            return "review_tool"
        if needs_confirm(check):
            return "confirm_tool"
        return "tools" if check["approved"] else "decline_tool"

    def after_review_tool(state: AgentState) -> str:
        return "tools" if state["tool_checks"][-1]["approved"] else "decline_tool"

    def back_to_route(state: AgentState) -> str:
        return state["decision"]["route"]

    graph.add_node("assess", assess)
    graph.add_node("clarify", clarify)
    for name, route in scenario.routes.items():
        graph.add_node(name, intake_node(route) if route.fields else route_node(route))
        if route_tools[name]:
            graph.add_conditional_edges(name, after_route, ["verify_tool", reply_done])
        else:
            graph.add_edge(name, reply_done)
    graph.add_edge(START, "assess")
    graph.add_edge("clarify", END)

    route_targets = [*scenario.routes, "clarify"]
    if review:
        graph.add_node("review_route", review_route)
        graph.add_node("review_reply", review_reply)
        graph.add_conditional_edges("assess", after_assess_in_review, ["review_route", *route_targets])
        graph.add_conditional_edges("review_route", after_assess, route_targets)
        graph.add_edge("review_reply", END)
    else:
        graph.add_conditional_edges("assess", after_assess, route_targets)

    if all_tools:
        with_tools = [name for name, tools in route_tools.items() if tools]
        graph.add_node("verify_tool", verify_tool)
        graph.add_node("decline_tool", decline_tool)
        graph.add_node("tools", ToolNode(list(all_tools.values()), handle_tool_errors=True))
        if review:
            graph.add_node("review_tool", review_tool)
            graph.add_conditional_edges("verify_tool", after_verify, ["review_tool", "tools", "decline_tool"])
            graph.add_conditional_edges("review_tool", after_review_tool, ["tools", "decline_tool"])
        else:
            graph.add_node("confirm_tool", confirm_tool)
            graph.add_conditional_edges("verify_tool", after_verify, ["confirm_tool", "tools", "decline_tool"])
            graph.add_conditional_edges("confirm_tool", after_review_tool, ["tools", "decline_tool"])
        graph.add_node("settle", settle)
        graph.add_edge("tools", "settle")
        graph.add_conditional_edges("settle", back_to_route, with_tools)
        graph.add_conditional_edges("decline_tool", back_to_route, with_tools)

    # Pausing needs a checkpointer; it only has to span one turn (see chat.run_turn)
    pauses = review or bool(scenario.confirm_tools)
    return graph.compile(checkpointer=InMemorySaver() if pauses else None)
