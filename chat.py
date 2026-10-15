"""Terminal chat for a scenario. Run mode by default; --review pauses for a human verdict on each decision."""
import os
# Set before huggingface_hub is imported, which reads it at import time
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# pylint: disable=wrong-import-position

import argparse
import sys
from urllib.parse import urlparse
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from clm_router import ClmError, make_router
from generator import GeneratorError, make_generator
from graph import build_graph, describe_calls, signals_line
from record import append_record, load_records, locked_lines, record_key, touch_record
from scenario import ScenarioError, load_scenario
from tools import load_catalogue, load_tools, resolve_tools


def format_decision(decision: dict, scenario) -> str:
    """CLM's routing table, confidence and extra-question signals for one turn."""
    lines = ["  CLM routing:"]
    for name, probability in sorted(decision["probabilities"].items(), key=lambda item: -item[1]):
        lines.append(f"    {name:<24}{probability:>8.2%}")
    lines.append(f"    confidence {decision['confidence']:.2f} (threshold {scenario.confidence_threshold:.2f})")
    signals = signals_line(decision["answers"]).strip()
    if signals:
        lines.append(f"  {signals}")
    if decision.get("rules"):
        lines.append("  rules applied: " + "; ".join(decision["rules"]))
    return "\n".join(lines)


def format_outcome(decision: dict, scenario) -> str:
    """Where the turn went and why."""
    if decision["source"] == "intake":
        return (f"  -> {decision['route']} (continuing; CLM said {decision['clm_route']}; "
                f"type 'new' to start over)")
    if decision["source"] == "reviewed":
        return (f"  -> {decision['route']} (reviewed before, similarity {decision['similarity']:.2f}; "
                f"CLM said {decision['clm_route']})")
    if decision["confidence"] >= scenario.confidence_threshold:
        return f"  -> {decision['route']}"
    return "  -> clarify (confidence below threshold)"


def format_tool_check(check: dict) -> str:
    """One proposed tool call and the verdict on it."""
    calls = describe_calls(check["calls"]).replace("- ", "", 1).replace("\n- ", "; ")
    if check["source"] == "unknown_tool":
        verdict = "declined: tool not available on this route"
    else:
        approved = check.get("clm_approved", check["approved"]) if check["source"] == "user" else check["approved"]
        verdict = f"CLM {check['p']:.0%} -> {'run' if approved else 'declined'}"
    return f"  tool call: {calls}\n    {verdict}"


SESSION_KEYS = ("messages", "draft", "draft_route", "draft_start", "follow_up")


def new_session() -> dict:
    """What the loop carries from turn to turn: the conversation and any request being drafted."""
    return {"messages": [], "draft": {}, "draft_route": None, "draft_start": 0, "follow_up": False}


def format_intake(out: dict) -> str:
    """How far along an open intake is; empty when none is open."""
    progress = out.get("intake")
    if not progress:
        return ""
    missing = f" (missing: {', '.join(progress['missing'])})" if progress["missing"] else " (complete)"
    return f"  fields: {progress['filled']} of {progress['total']}{missing}"


def format_draft(session: dict, scenario) -> str:
    """The request being drafted, field by field."""
    route = scenario.routes.get(session.get("draft_route") or "")
    if route is None:
        return "  no request in progress"
    lines = [f"  {route.name} draft:"]
    lines += [f"    {ask}: {session['draft'].get(name, '(still needed)')}" for name, ask in route.fields.items()]
    return "\n".join(lines)


def format_turn(out: dict, scenario) -> str:
    """Everything run mode prints for a turn: decision, outcome, tool checks, reply."""
    decision = out["decision"]
    reply = str(out["messages"][-1].content) or "(no reply)"
    lines = [format_decision(decision, scenario), format_outcome(decision, scenario)]
    # A check the user confirmed was already shown at the prompt
    lines += [format_tool_check(check) for check in out.get("tool_checks", []) if check["source"] != "user"]
    if format_intake(out):
        lines.append(format_intake(out))
    lines.append(f"agent> {reply}")
    return "\n".join(lines)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def format_routes(scenario) -> str:
    """The scenario's routes with their descriptions and example messages."""
    lines = []
    for route in scenario.routes.values():
        lines.append(f"  {route.name}: {route.description}")
        lines += [f'      e.g. "{example}"' for example in route.examples]
    return "\n".join(lines)


def format_help(scenario, review: bool) -> str:
    """The text shown for `help`."""
    if review:
        mode = (f"Mode: review. For each message you confirm or correct the route, approve tool calls and "
                f"rate the reply.\n  Verdicts are saved to {scenario.record_path.name}; the route you choose is "
                f"reused when the same message comes up again.")
    else:
        mode = (f"Mode: run. Replies without pausing, reusing the routes chosen in review mode "
                f"({scenario.record_path.name}).\n  Start with --review to teach it.")
    return "\n".join([
        f"This agent handles {_count(len(scenario.routes), 'kind')} of request ({scenario.path.name}):",
        format_routes(scenario),
        mode,
        "Commands: help or ? (this text), draft (show the request being drafted), new (discard it), "
        "exit or quit (or Ctrl-D)",
    ])


def _where(url: str) -> str:
    local = urlparse(url).hostname in ("localhost", "127.0.0.1", "::1")
    return f"sent to {url} ({'on this computer' if local else 'ANOTHER MACHINE'})"


def format_data_notice(scenario, review: bool, encrypted: bool) -> str:
    """Where the text of a conversation goes with this scenario: each model, and the record."""
    clm = _where(scenario.clm_url) if scenario.clm_backend == "server" else "runs in this process"
    if scenario.generator_backend == "openai":
        generator = _where(scenario.generator_base_url)
    elif scenario.generator_backend == "ollama":
        generator = "sent to the Ollama server this computer is set up to use"
    else:
        generator = "runs in this process"
    if review:
        record = (f"review mode saves conversation text to {scenario.record_path} "
                  f"({'encrypted' if encrypted else 'not encrypted'})")
    else:
        record = "nothing is saved in run mode"
    lines = ["Where conversation text goes:", f"  CLM: {clm}", f"  language model: {generator}",
             "  tools: wherever the scenario's tools send it (see [tools.settings])", f"  record: {record}"]
    tracing = (os.environ.get(name, "").lower() for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"))
    if any(value in ("1", "true", "yes") for value in tracing):
        lines.append("  tracing: LangSmith tracing is switched on in your environment, so prompts are ALSO sent "
                     "to LangSmith")
    return "\n".join(lines)


def make_recorder(scenario, router, write=print):
    """The review-mode record callback: save the verdict, apply it at once, and say what was stored."""
    key = record_key(scenario)

    def record(entry: dict) -> None:
        append_record(scenario.record_path, {**entry, "scenario_sha": scenario.sha}, key=key)
        if entry["kind"] != "route":
            return
        count = router.load_verdicts(load_records(scenario.record_path, key=key)[0], scenario)
        clm, human = entry["clm"]["route"], entry["human"]
        verdict = f"{clm} confirmed" if clm == human else f"{clm} -> {human}"
        write(f"  recorded: {verdict} ({_count(count, 'reviewed route')} on file)")
    return record


def _yes_no(read, prompt: str) -> bool:
    while True:
        answer = read(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def ask_human(payload: dict, scenario, read, write) -> dict:
    """Show one paused decision and return the reviewer's verdict in the shape the graph expects."""
    if payload["kind"] == "route":
        decision, names = payload["decision"], payload["routes"]
        write(format_decision(decision, scenario))
        if decision["source"] == "reviewed":
            write(f"  reviewed before (similarity {decision['similarity']:.2f}): {decision['route']} "
                  f"(CLM alone said {decision['clm_route']})")
        while True:
            answer = read(f"route? [Enter = {decision['route']}, a route name, or ? to list them] ").strip()
            if not answer:
                return {"route": decision["route"]}
            if answer == "?":
                write(format_routes(scenario))
                continue
            matches = [name for name in names if name.lower().startswith(answer.lower())]
            if answer in names:
                return {"route": answer}
            if len(matches) == 1:
                return {"route": matches[0]}
            write("  routes: " + ", ".join(names))
    if payload["kind"] == "confirm":
        write(format_tool_check(payload["check"]))
        return {"approved": _yes_no(read, "go ahead? [y/n] ")}
    if payload["kind"] == "tool":
        write(format_tool_check(payload["check"]))
        return {"approved": _yes_no(read, "run this tool call? [y/n] ")}
    for check in payload.get("tool_checks", []):
        write(format_tool_check(check))
    write(f"agent> {payload['reply'] or '(no reply)'}")
    acceptable = _yes_no(read, "acceptable? [y/n] ")
    return {"acceptable": acceptable, "note": read("note (optional): ").strip()}


def start_turn(graph, session: dict, user_text: str) -> tuple[dict, dict]:
    """Begin a user turn: -> (the graph's output, the config that resumes the turn if it paused)."""
    # A new thread per turn: the caller owns the session, and the checkpointer only has to
    # remember a turn while it is paused. Reusing a thread would add the history twice.
    config = {"configurable": {"thread_id": uuid4().hex}}
    inputs = {**session, "messages": session["messages"] + [HumanMessage(content=user_text)]}
    return graph.invoke(inputs, config), config


def resume_turn(graph, config: dict, answer: dict) -> dict:
    """Continue a paused turn with a person's answer."""
    return graph.invoke(Command(resume=answer), config)


def paused(out: dict) -> dict | None:
    """What a turn is waiting for a person to answer, or None when the turn is complete."""
    interrupts = out.get("__interrupt__")
    return interrupts[0].value if interrupts else None


def run_turn(graph, session: dict, user_text: str, ask=None) -> dict:
    """Run one user turn to completion, calling `ask` each time the graph pauses for a person."""
    out, config = start_turn(graph, session, user_text)
    while paused(out) is not None:
        out = resume_turn(graph, config, ask(paused(out)))
    return out


def next_session(out: dict) -> dict:
    """The session to carry into the next turn, from a completed turn's output."""
    return {key: out.get(key, default) for key, default in new_session().items()}


def reset_draft(session: dict) -> dict:
    """The session after `new`: the conversation is kept, the request being drafted is not."""
    return {**new_session(), "messages": session["messages"], "draft_start": len(session["messages"])}


def chat_loop(graph, scenario, review: bool = False, read=input, write=print) -> None:
    """Read messages until exit; `read` and `write` are replaceable for tests."""
    session = new_session()

    def ask(payload):
        return ask_human(payload, scenario, read, write)

    while True:
        try:
            text = read("you> ").strip()
        except EOFError:
            break
        if text.lower() in ("exit", "quit"):
            break
        if not text:
            continue
        if text.lower() in ("help", "?"):
            write(format_help(scenario, review))
            continue
        if text.lower() == "draft":
            write(format_draft(session, scenario))
            continue
        if text.lower() == "new":
            session = reset_draft(session)
            write("  started over")
            continue
        try:
            out = run_turn(graph, session, text, ask)
        except EOFError:
            break
        except (GeneratorError, ClmError) as error:
            write(f"error: {error}")
            continue
        session = next_session(out)
        if not review:
            write(format_turn(out, scenario))
        elif format_intake(out):
            write(format_intake(out))


class StartupError(Exception):
    """Start-up cannot continue; the message says why."""


def load_agent(path: str, review: bool = False):  # pylint: disable=too-many-locals
    """What every front end does before the first message: -> (scenario, graph), saying what it loads."""
    try:
        scenario = load_scenario(path)
        registry = load_tools(scenario)
        resolve_tools(scenario, registry)
        catalogue = load_catalogue(scenario, registry)
        key = record_key(scenario)
    except ScenarioError as error:
        raise StartupError(str(error)) from error

    # Check the record before the models load, so a bad path costs seconds and not a lost verdict
    action = "write" if review else "read"
    try:
        locked = locked_lines(scenario.record_path, key)
        if review and locked:
            raise StartupError(
                f"{_count(locked, 'line')} of the record cannot be read with the current key setting. "
                f"Reviewing now would leave a record no single key can read. Restore the key it was written "
                f"with, or delete it with `python record.py purge {path}`.")
        if review:
            touch_record(scenario.record_path)
        records, skipped = load_records(scenario.record_path, key=key)
    except OSError as error:
        raise StartupError(f"cannot {action} the record {scenario.record_path}: {error}") from error

    print(format_data_notice(scenario, review, encrypted=key is not None))
    print(f"Loading CLM ({scenario.clm_model_id}) and generator ({scenario.generator_model})...")
    router, clm_note = make_router(scenario)
    print(f"CLM is using {clm_note}.")
    generator = make_generator(scenario)

    count = router.load_verdicts(records, scenario)
    note = f" ({skipped} unreadable lines skipped; a missing or different record key would cause this)" \
        if skipped else ""
    mode = "Review mode" if review else "Run mode"
    print(f"{mode}: {_count(count, 'reviewed route')} loaded from {scenario.record_path.name}{note}")

    if catalogue is not None:
        print(f"Catalogue: {len(catalogue)} entries from {scenario.catalogue.tool}.")
    record = make_recorder(scenario, router) if review else None
    graph = build_graph(scenario, router, generator, registry=registry, review=review, record=record,
                        catalogue=catalogue)
    print(f"Ready. This agent handles: {', '.join(scenario.routes)}.")
    return scenario, graph


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Chat with a scenario-driven local agent.")
    parser.add_argument("scenario", help="path to the scenario file")
    parser.add_argument("--review", action="store_true",
                        help="pause at each decision for a human verdict and record it")
    args = parser.parse_args(argv)
    try:
        scenario, graph = load_agent(args.scenario, args.review)
    except StartupError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print("Type a message, 'help' for what it can do, 'exit' to quit.")
    chat_loop(graph, scenario, review=args.review)
    return 0


if __name__ == "__main__":
    sys.exit(main())
