"""Write labelled example messages for each route of a scenario, using the scenario's generator.

    uv run python synthesize.py <scenario>

The examples are what `train.py` learns from. They are the generator's idea of how people write,
so read some of them: the file is plain text, one example per line, and can be edited.
"""
import os
# Set before huggingface_hub is imported, which reads it at import time
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# pylint: disable=wrong-import-position

import argparse
import json
import re
import sys

from langchain_core.messages import HumanMessage

from dataset import examples_path, save_examples
from generator import GeneratorError, make_generator
from scenario import load_for_command

STYLES = (
    "one short sentence, straight to the point",
    "polite and detailed, two or three sentences",
    "vague, the way someone writes when they are not sure what to ask for",
    "informal, written in a hurry, with lower case and abbreviations",
    "says something about who the writer is and why they are asking",
    "mentions a deadline or how urgent it is",
    "phrased as a question",
    "starts with some background before getting to the request",
)
PROMPT = (
    "Write {count} different {messages}.\n"
    "Every message must be this kind of request: {description}\n"
    "{examples}"
    "It must not be any of these other kinds:\n{others}\n"
    "Style: {style}.\n"
    "Vary the subject from message to message. Use invented details only.\n"
    "Return only a JSON list of strings."
)


def build_prompt(scenario, route, style: str, count: int) -> str:
    """The request to the generator for one batch of messages for one route."""
    others = "\n".join(f"- {other.description}" for other in scenario.routes.values() if other is not route)
    examples = f"For example: {'; '.join(route.examples)}\n" if route.examples else ""
    return PROMPT.format(count=count, messages=scenario.synthesize_messages, description=route.description,
                         examples=examples, others=others, style=style)


def _strings(value) -> list[str]:
    """Every non-empty string in a parsed JSON value, however lists and objects are nested."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        value = list(value.values())
    return [text for item in value for text in _strings(item)] if isinstance(value, list) else []


def _json_list(text: str) -> list[str]:
    """The strings in the outermost [...] of the text, forgiving a trailing comma; [] if it is not JSON."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return _strings(json.loads(re.sub(r",\s*\]", "]", match.group(0))))
    except json.JSONDecodeError:
        return []


def parse_messages(text: str) -> list[str]:
    """The messages in the generator's output: a JSON list of strings, or failing that one per line."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    found = _json_list(text)
    if found:
        return found
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and _json_list(line):
            found += _json_list(line)
            continue
        # Output cut off mid-list leaves brackets and half-quoted strings; headings end with a colon
        cut_off = line.count('"') % 2 == 1
        line = re.sub(r"^(?:[-*]|\d+[.)])\s*", "", line).rstrip(",").strip().strip('"')
        if not cut_off and re.search(r"[A-Za-z]{2}", line) and not line.endswith(":") and not line.startswith("`"):
            found.append(line)
    return found


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def synthesize(scenario, generator, path, per_route: int = 64, per_call: int = 8, write=print) -> int:
    """Ask for messages route by route, cycling through the styles, until each route has enough."""
    rows = []
    for route in scenario.routes.values():
        seen, call, idle = set(), 0, 0
        # Two calls in a row with nothing new means the generator has run out of variety
        while len(seen) < per_route and idle < 2:
            style = STYLES[call % len(STYLES)]
            reply = generator.generate(build_prompt(scenario, route, style, per_call),
                                       [HumanMessage(content=f"Batch {call + 1}.")])
            fresh = [m for m in dict.fromkeys(parse_messages(str(reply.content))) if m not in seen]
            fresh = fresh[:per_route - len(seen)]
            idle = 0 if fresh else idle + 1
            seen.update(fresh)
            rows += [{"text": message, "route": route.name, "style": style} for message in fresh]
            call += 1
        write(f"  {route.name:<24}{len(seen):>4} examples")
    save_examples(path, rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Write labelled example messages for a scenario's routes.")
    parser.add_argument("scenario", help="path to the scenario file")
    parser.add_argument("--per-route", type=int, default=64, help="examples to write for each route")
    args = parser.parse_args(argv)
    scenario = load_for_command(args.scenario)
    if scenario is None:
        return 2
    path = examples_path(scenario)
    if path.exists():
        print(f"error: {path} already exists. Move or delete it to write a new set.", file=sys.stderr)
        return 2
    print(f"Writing examples with {scenario.generator_model}...")
    try:
        count = synthesize(scenario, make_generator(scenario), path, per_route=args.per_route)
    except GeneratorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"{count} examples written to {path}. Read some before training on them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
