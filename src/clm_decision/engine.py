"""Run an environment's queries, decisions, calculations and scenarios, recording every step.

Every number a decision rests on passes through here: a query's rows, a rule's inputs and the rules that fired, a
calculation's result. The owner's branch function asks for them through `Steps`, so the engine carries the numbers
and the agent only chooses which branches to run. A branch is rejected when a limit says not allowed, and fails
when a step fails or a limit was never checked, so no branch can skip a limit unnoticed.
"""
from __future__ import annotations

import itertools
import math

from . import tracing
from .dmn import DmnError, evaluate
from .environment import Environment
from .scenario import StepError

_CHECKS = {"string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
           "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
           "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
           "list": lambda v: isinstance(v, list), "any": lambda _: True}
_ARTICLES = {"string": "text", "boolean": "true or false", "integer": "a whole number", "number": "a number",
             "list": "a list"}


def _check(what: str, declared: dict[str, str], given: dict) -> None:
    missing = [name for name in declared if name not in given]
    if missing:
        raise StepError(f"{what}: missing parameter {', '.join(missing)}")
    unknown = [name for name in given if name not in declared]
    if unknown:
        raise StepError(f"{what}: unknown parameter {', '.join(unknown)}")
    for name, kind in declared.items():
        if not _CHECKS[kind](given[name]):
            raise StepError(f"{what}: {name} must be {_ARTICLES[kind]}, not {given[name]!r}")


def _message(error: Exception) -> str:
    """An error as one line: the engine's own as they are, anything else with its type."""
    return str(error) if isinstance(error, (StepError, DmnError)) else f"{type(error).__name__}: {error}"


def _part(parts: dict, kind: str, name: str):
    if name not in parts:
        known = ", ".join(sorted(parts)) or "none"
        raise StepError(f"no {kind} '{name}' (the environment has: {known})")
    return parts[name]


class Steps:
    """What a branch function, or a single call, asks of the environment, each answer recorded in `steps`."""

    def __init__(self, env: Environment):
        self.env = env
        self.steps: list[dict] = []

    def query(self, name: str, **params) -> list[dict]:
        """A named query's rows."""
        query = _part(self.env.queries, "query", name)
        _check(f"query {name}", query.params, params)
        tracing.add("query")
        with tracing.timed("query_ms"):
            rows = self.env.connector.run(query, params)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise StepError(f"query {name}: its function must return a list of dicts")
        self.steps.append({"kind": "query", "name": name, "inputs": params, "result": rows})
        return rows

    def decide(self, name: str, **inputs) -> dict:
        """A decision's outputs; a DMN decision also records the rules that fired."""
        decision = _part(self.env.decisions, "decision", name)
        _check(f"decision {name}", decision.params, inputs)
        tracing.add("decision")
        try:
            with tracing.timed("decision_ms"):
                if decision.table is not None:
                    result = evaluate(decision.table, inputs)
                    outputs, fired = result["outputs"], result["fired"]
                else:
                    outputs, fired = getattr(self.env.functions, decision.function)(**inputs), []
            if decision.table is None and not isinstance(outputs, dict):
                raise StepError(f"decision {name}: its function must return a dict")
        except Exception as error:
            # Recorded even if the owner's code catches it, so a limit that failed cannot pass unnoticed
            self.steps.append({"kind": "decision", "name": name, "inputs": inputs, "result": None, "fired": [],
                               "error": _message(error)})
            raise
        self.steps.append({"kind": "decision", "name": name, "inputs": inputs, "result": outputs, "fired": fired})
        return outputs

    def judge(self, name: str, text: str) -> dict:
        """CLM's judgement of the text: {"probability"}, a number from 0 to 1."""
        judge = _part(self.env.judges, "judge", name)
        if not isinstance(text, str) or not text.strip():
            raise StepError(f"judge {name}: there is no text to judge")
        tracing.add("judgement")
        try:
            with tracing.timed("judgement_ms"):
                result = {"probability": self.env.clm.probability(judge, text)}
        except Exception as error:
            self.steps.append({"kind": "judgement", "name": name, "inputs": {"text": text}, "result": None,
                               "error": _message(error)})
            raise
        self.steps.append({"kind": "judgement", "name": name, "inputs": {"text": text}, "result": result})
        return result

    def calculate(self, name: str, **inputs) -> dict:
        """A calculation's result."""
        calculation = _part(self.env.calculations, "calculation", name)
        _check(f"calculation {name}", calculation.params, inputs)
        tracing.add("calculation")
        with tracing.timed("calculation_ms"):
            result = getattr(self.env.functions, calculation.function)(**inputs)
        if not isinstance(result, dict):
            raise StepError(f"calculation {name}: its function must return a dict")
        self.steps.append({"kind": "calculation", "name": name, "inputs": inputs, "result": result})
        return result


def run_query(env: Environment, name: str, params: dict) -> dict:
    """-> {"rows", "steps"}."""
    steps = Steps(env)
    return {"rows": steps.query(name, **params), "steps": steps.steps}


def run_decision(env: Environment, name: str, inputs: dict) -> dict:
    """-> {"outputs", "fired", "steps"}."""
    steps = Steps(env)
    outputs = steps.decide(name, **inputs)
    return {"outputs": outputs, "fired": steps.steps[-1]["fired"], "steps": steps.steps}


def run_judgement(env: Environment, name: str, inputs: dict) -> dict:
    """-> {"result", "steps"}."""
    _check(f"judge {name}", {"text": "string"}, inputs)
    steps = Steps(env)
    return {"result": steps.judge(name, inputs["text"]), "steps": steps.steps}


def run_calculation(env: Environment, name: str, inputs: dict) -> dict:
    """-> {"result", "steps"}."""
    steps = Steps(env)
    return {"result": steps.calculate(name, **inputs), "steps": steps.steps}


def _branches(scenario, branches: list[dict] | None) -> list[dict]:
    """The branches to run, every combination by default; raises StepError for any value the owner did not allow."""
    if branches is None:
        return [dict(zip(scenario.levers, values)) for values in itertools.product(*scenario.levers.values())]
    if not branches:
        raise StepError(f"scenario {scenario.name}: choose at least one branch, or leave branches out to run them all")
    for index, branch in enumerate(branches):
        if not isinstance(branch, dict):
            raise StepError(f"scenario {scenario.name}: each branch is a table of lever values")
        for lever, value in branch.items():
            if lever not in scenario.levers:
                raise StepError(f"scenario {scenario.name}: unknown lever '{lever}' "
                                f"(levers: {', '.join(scenario.levers)})")
            allowed = list(scenario.levers[lever])
            # True == 1 in Python, so a value matches only an allowed value of the same kind
            if not any(value == choice and isinstance(value, bool) == isinstance(choice, bool) for choice in allowed):
                raise StepError(f"scenario {scenario.name}: {lever} = {value!r} is not allowed; "
                                f"choose from {allowed}")
        missing = [lever for lever in scenario.levers if lever not in branch]
        if missing:
            raise StepError(f"scenario {scenario.name}: a branch needs a value for {', '.join(missing)}")
        if branch in branches[:index]:
            raise StepError(f"scenario {scenario.name}: the branch {branch} is given twice")
    return [dict(branch) for branch in branches]


def _limits(scenario, steps: list[dict]) -> list[dict]:
    """The limit decisions that said not allowed; raises StepError for a limit never checked or badly answered."""
    rejected = []
    for limit in scenario.limits:
        checks = [step for step in steps if step["kind"] == "decision" and step["name"] == limit]
        if not checks:
            raise StepError(f"limit {limit} was not checked")
        for check in checks:
            if check.get("error"):
                raise StepError(f"limit {limit} failed: {check['error']}")
            outputs = check["result"]
            if not isinstance(outputs, dict):
                raise StepError(f"limit {limit}: its outputs must be a dict with allowed and reason")
            if not isinstance(outputs.get("allowed"), bool) or not isinstance(outputs.get("reason"), str):
                raise StepError(f"limit {limit}: its outputs need allowed (true or false) and reason (text)")
            if not outputs["allowed"]:
                rejected.append({"decision": limit, "reason": outputs["reason"], "inputs": check["inputs"]})
    return rejected


def _run_branch(env: Environment, scenario, params: dict, levers: dict) -> dict:
    steps = Steps(env)
    branch = {"levers": levers, "status": "failed", "figures": {}, "objective": None, "rejected_by": [],
              "error": None, "steps": steps.steps}
    try:
        figures = getattr(env.functions, scenario.branch)(steps, dict(params), dict(levers))
        if not isinstance(figures, dict):
            raise StepError(f"scenario {scenario.name}: the branch function must return a dict of figures")
        branch["figures"] = figures
        objective = figures.get(scenario.objective)
        if not isinstance(objective, (int, float)) or isinstance(objective, bool):
            raise StepError(f"scenario {scenario.name}: the figures have no number for {scenario.objective}")
        if not math.isfinite(objective):
            raise StepError(f"scenario {scenario.name}: the figures' {scenario.objective} is not a finite number")
        branch["objective"] = objective
        branch["rejected_by"] = _limits(scenario, steps.steps)
        branch["status"] = "rejected" if branch["rejected_by"] else "kept"
    # The owner's code can fail in any way; one branch failing should not stop the others
    except Exception as error:  # pylint: disable=broad-exception-caught
        branch["error"] = _message(error)
    return branch


def run_scenario(env: Environment, name: str, params: dict, branches: list[dict] | None = None) -> dict:
    """Run each branch (every combination of the levers by default) and rank the ones no limit rejected."""
    scenario = _part(env.scenarios, "scenario", name)
    _check(f"scenario {name}", scenario.params, params)
    results = [_run_branch(env, scenario, params, levers) for levers in _branches(scenario, branches)]
    kept = sorted((branch for branch in results if branch["status"] == "kept"),
                  key=lambda branch: branch["objective"], reverse=scenario.maximise)
    ranking = [branch["levers"] for branch in kept]
    return {"scenario": name, "objective": scenario.objective, "maximise": scenario.maximise, "branches": results,
            "ranking": ranking, "best": ranking[0] if ranking else None}
