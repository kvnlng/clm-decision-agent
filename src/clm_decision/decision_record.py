"""The decision record: what a financial officer, or anyone checking an answer, reads instead of the chat.

It is built only from the tools' results, in the order they were called, so the same calls give the same record, and
it works for a server reached by URL as well as one started from a file. It holds the question; what the person
asking supplied; the rules that applied, with what each says, why it exists, its owner and version; every fact and
estimate used, an estimate with its range, source and date; each scenario's branches with the limits that rejected
them, the warnings at an estimate's low and high, the headroom and the approvals needed; what is still to ask; and
any call that failed. The agent's own explanation is added beside it, marked as not checked.
"""
from __future__ import annotations

import json

_RANGE_KEYS = ("low", "central", "high", "source", "as_of")


def _information(record: dict, steps: list[dict], descriptions: dict) -> None:
    """Add each query a run used, once per query and inputs, with its rows; an estimate's with their ranges and
    sources."""
    for step in steps:
        if step["kind"] != "query" or step.get("case", "central") != "central":
            continue
        item = {"query": step["name"], "kind": step.get("information", "fact"), "inputs": step["inputs"],
                "description": descriptions.get(step["name"], "")}
        item["rows"] = [{key: row[key] for key in _RANGE_KEYS} for row in step["result"]] \
            if item["kind"] == "estimate" else step["result"]
        if not any(seen["query"] == item["query"] and seen["inputs"] == item["inputs"]
                   for seen in record["information"]):
            record["information"].append(item)


def _scenario(result: dict) -> dict:
    keys = ("levers", "status", "figures", "rejected_by", "warnings", "headroom", "approvals", "error")
    return {"scenario": result["scenario"], "objective": result["objective"], "maximise": result["maximise"],
            "best": result["best"], "ranking": result["ranking"],
            "branches": [{key: branch.get(key) for key in keys} for branch in result["branches"]]}


def build(question: str, calls: list[dict]) -> dict:
    """The record of one answer, from the calls clm-agent made: each {"tool", "arguments", "result", "error"}."""
    record: dict = {"question": question, "supplied": {}, "rules": {}, "information": [], "scenarios": [],
                    "to_ask": [], "failed_calls": []}
    asked: list[dict] = []
    for call in calls:
        tool, result = call["tool"], call["result"]
        if call["error"] or not isinstance(result, dict):
            record["failed_calls"].append({"tool": tool, "error": call["error"] or "no result"})
            continue
        if tool.startswith("scenario_"):
            record["supplied"].update(result.get("supplied", {}))
            record["rules"].update(result.get("rules", {}))
            descriptions = {name: item["description"] for name, item in result.get("information", {}).items()}
            for branch in result["branches"]:
                _information(record, branch["steps"], descriptions)
            record["scenarios"].append(_scenario(result))
        elif tool == "rules":
            record["rules"].update({rule["name"]: rule for rule in result["rules"]})
            asked += [item for item in result["to_ask"] if item not in asked]
        elif tool == "rule":
            record["rules"][result["name"]] = {key: value for key, value in result.items() if key != "rows"}
        elif "steps" in result:
            _information(record, result["steps"], {})
    # Supplied information is named after the scenario parameter that carries it
    record["to_ask"] = [item for item in asked if item["name"] not in record["supplied"]]
    return record


def _figures(figures: dict) -> str:
    return ", ".join(f"{name} {value}" for name, value in figures.items())


def _branch(branch: dict) -> list[str]:
    levers = ", ".join(f"{lever} = {value}" for lever, value in branch["levers"].items())
    lines = [f"- **{levers}**: {branch['status']}" + (f": {_figures(branch['figures'])}" if branch["figures"] else "")]
    if branch.get("error"):
        lines.append(f"  - failed: {branch['error']}")
    lines += [f"  - rejected by {item['decision']}: {item['reason']}" for item in branch.get("rejected_by") or []]
    lines += [f"  - warning: at {item['estimate']}'s {item['case']} estimate, "
              f"{item['limit'] or 'the branch'}: {item['reason']}" for item in branch.get("warnings") or []]
    for item in branch.get("headroom") or []:
        if item["change"] is None:
            lines.append(f"  - headroom {item['estimate']}: {item['direction']}: no limit fails within the range")
        else:
            lines.append(f"  - headroom {item['estimate']}: {item['direction']} {item['change']:+.1%} "
                         f"before {item['limit'] or 'the branch fails'}")
    lines += [f"  - needs approval from {item['who']} ({item['decision']}): {item['reason']}"
              for item in branch.get("approvals") or []]
    return lines


def markdown(record: dict, answer: str, unsupported: list[str]) -> str:
    """The record as a page to read: the checked record first, then the agent's explanation, marked as such."""
    out = ["# Decision record", "", "## Question", "", record["question"], "", "## Supplied by the asker", ""]
    out += [f"- {name}: {value}" for name, value in record["supplied"].items()] or ["Nothing."]
    out += ["", "## Rules applied", ""]
    for rule in record["rules"].values():
        out.append(f"- **{rule['name']}** (owner {rule['owner']}, version {rule['version']}): {rule['statement']} "
                   f"Why: {rule['reason']}")
    if not record["rules"]:
        out.append("No rule metadata was returned.")
    out += ["", "## Information", ""]
    for item in record["information"]:
        inputs = json.dumps(item["inputs"])
        if item["kind"] == "estimate":
            for row in item["rows"]:
                out.append(f"- {item['query']} {inputs} (estimate): {row['central']}, range {row['low']} to "
                           f"{row['high']}; {row['source']}, {row['as_of']}")
        else:
            out.append(f"- {item['query']} {inputs} (fact): {item['description']}")
            out += [f"  - {json.dumps(row)}" for row in item["rows"]]
    out += ["", "## Scenarios", ""]
    for scenario in record["scenarios"]:
        out.append(f"**{scenario['scenario']}**, by {scenario['objective']} "
                   f"({'highest' if scenario['maximise'] else 'lowest'} first); best: {scenario['best']}")
        for branch in scenario["branches"]:
            out += _branch(branch)
        out.append("")
    if not record["scenarios"]:
        out += ["No scenario was run.", ""]
    if record["to_ask"]:
        out += ["## Still to ask", ""] + [f"- {item['name']}: {item['description']}" for item in record["to_ask"]]
        out.append("")
    if record["failed_calls"]:
        out += ["## Calls that failed", ""] + [f"- {item['tool']}: {item['error']}" for item in record["failed_calls"]]
        out.append("")
    out += ["## The agent's explanation (not checked)", "", answer or "(no answer)", ""]
    if unsupported:
        out += ["## Figures no tool returned", "", ", ".join(unsupported), ""]
    return "\n".join(out)
