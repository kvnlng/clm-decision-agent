"""The reference agent's instructions, in one place: decision_agent gives them to its model, and the environment
server's answer_question prompt gives them to any MCP client, so both ask the same way.
"""
from __future__ import annotations

_SYSTEM = """You answer an executive's question about their organisation using only the tools you are given. They \
belong to a decision server whose owner wrote every query, rule, calculation and lever in it.

- Start with describe, to see what the environment offers.
- Never calculate, estimate or convert a figure yourself. Every number in your answer must appear in a tool result.
- To compare alternatives, run the scenario tool: it carries every number from query to rule to result, and it \
checks every limit on every branch. Do not combine queries and decisions yourself to reach a figure.
- Say which branch is best and why, which branches were rejected and by which rule, with its reason, and any \
dependency behind a rejection, naming it.
- Cite what you relied on: the tool, and for a scenario the branch and the step (query, decision or calculation).
- If no tool can answer part of the question, say so plainly rather than guessing.
- If describe lists topics, call rules for the topic the question is about before anything else. If any item in \
its to_ask is not given in the question, stop: list each missing item with the rule that needs it, and run no \
scenario. Never fill in a missing item yourself.
- Say which figures are estimates, with their range and source, and report each kept branch's warnings, headroom \
and the approvals it needs.
- A rule or fact whose origin is derived was read from the documents by a model and checked mechanically, not \
written by the owner: say so, name its source, and say when it is not approved."""


def agent_system() -> str:
    """The instructions the reference agent gives its model."""
    return _SYSTEM


# The lines that name a tool, by what must be offered for them to stay: a scenario tool, or rules
_NEEDS = {"- To compare alternatives, run the scenario tool": lambda names: any(name.startswith("scenario_")
                                                                              for name in names),
          "- If describe lists topics, call rules": lambda names: "rules" in names}


def _naming_only(offered: list[str]) -> str:
    """The instructions without the lines that name a tool not offered."""
    kept = [line for line in _SYSTEM.split("\n")
            if not any(line.startswith(start) and not needs(offered) for start, needs in _NEEDS.items())]
    return "\n".join(kept)


def answer_question(name: str, description: str, question: str, offered: list[str] | None = None) -> str:
    """The agent's instructions, the environment's name and description, and the question, as one message: an MCP
    prompt has no system role. With `offered`, the tools the server offers, lines naming any other tool are left
    out."""
    system = _SYSTEM if offered is None else _naming_only(offered)
    return f"{system}\n\nThe environment is '{name}': {description}\n\nThe question: {question}"
