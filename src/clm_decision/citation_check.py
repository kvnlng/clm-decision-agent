"""Check that every citation in an agent's answer is one the tools returned, as number_check.py does for figures.

Two kinds are checked. A **statute** is matched by its section, so "§15.50", "§ 15.50" and "Tex. Bus. & Com. Code Ann.
§ 15.50" all match a stored "Tex. Bus. & Com. Code § 15.50"; a subsection, "§15.51(c)", matches when the tools
returned that section's passage labelled "(c)", or a citation naming it. "Section 15.50", "§§ 15.50 and 15.51",
"820 ILCS 90/10" and "RCW 49.62.020" are read too. A **reporter citation**, "100 S.W.3d 1", is matched by volume,
reporter and page, ignoring the reporter's spaces and periods. A case name with no reporter citation after it
("Smith v. Jones") cannot be checked and is listed as such, never passed. Sections are compared without their code,
so two codes' sections with the same number would match each other.
"""
from __future__ import annotations

import re

_SECTION = r"\d+[A-Za-z]?(?:[.\-:/]\d+[A-Za-z]?)*"
_LABELS = r"(?:\(\w{1,4}\))*"
_STATUTE = re.compile(rf"(§§?|\bsec(?:tion\b|\.))\s*({_SECTION})({_LABELS})", re.IGNORECASE)
_MORE = re.compile(rf"\s*(?:,\s*and|,|and|&)\s*({_SECTION})({_LABELS})")
_NAMED = re.compile(rf"\b(?:\d+\s+ILCS|RCW)\s+({_SECTION})({_LABELS})")
_TOKEN = r"(?:[A-Z][A-Za-z]*\.?|\d+(?:d|th|st|nd|rd)\b)"
_REPORTER = re.compile(rf"\b(\d{{1,4}})\s+({_TOKEN}(?:\s?{_TOKEN}){{0,4}})\s+(\d{{1,5}})\b")
_PARTY = r"[A-Z][\w.'&-]*(?:\s+(?:[A-Z][\w.'&-]*|of|the|and|&))*"
_CASE_NAME = re.compile(rf"\b({_PARTY})\s+v\.\s+({_PARTY})")
# A case name's reporter citation follows it at once: "Name v. Name, 100 S.W.3d 1"
_FOLLOWED = re.compile(r",?\s*\d{1,4}\s")


def _label(labels: str) -> str | None:
    """The first subsection label, "(c)" of "(c)(1)"; None without one."""
    return labels[:labels.index(")") + 1].lower() if labels else None


def _statutes(text: str) -> list[tuple[str, str, str | None, int]]:
    """Each statute cited: (as written, section, subsection label, position)."""
    found = []
    for match in _STATUTE.finditer(text):
        found.append((match.group(0), match.group(2).lower(), _label(match.group(3)), match.start()))
        if match.group(1) == "§§":
            at = match.end()
            while more := _MORE.match(text, at):
                found.append((f"{more.group(1)}{more.group(2)}", more.group(1).lower(), _label(more.group(2)),
                              more.start(1)))
                at = more.end()
    found += [(m.group(0), m.group(1).lower(), _label(m.group(2)), m.start()) for m in _NAMED.finditer(text)]
    return found


def _reporters(text: str) -> list[tuple[str, str, int]]:
    """Each reporter citation: (as written, volume-reporter-page key, position)."""
    return [(m.group(0), f"{m.group(1)} {re.sub(r'[\s.]', '', m.group(2)).lower()} {m.group(3)}", m.start())
            for m in _REPORTER.finditer(text) if "." in m.group(2)]


def keys(text: str) -> set[tuple]:
    """What a citation's text cites: ("statute", section, label) and ("reporter", key) items."""
    return ({("statute", section, label) for _, section, label, _ in _statutes(text)}
            | {("reporter", key) for _, key, _ in _reporters(text)})


def _returned(result, found: set[tuple]) -> None:
    """Every citation in the tool results, at any depth, with the passage labels returned for it."""
    if isinstance(result, dict):
        citation = result.get("citation")
        if isinstance(citation, str):
            cited = keys(citation)
            found |= cited
            labels = [result["passage"]] if isinstance(result.get("passage"), str) else []
            labels += [p["label"] for p in result.get("passages") or [] if isinstance(p, dict) and "label" in p]
            found |= {("statute", item[1], label.lower()) for item in cited if item[0] == "statute"
                      for label in labels}
        for value in result.values():
            _returned(value, found)
    elif isinstance(result, (list, tuple)):
        for item in result:
            _returned(item, found)


def matches(written: str, citation: str, passage: str | None = None) -> bool:
    """Whether a citation as written in an answer is the stored citation, or its passage."""
    stored = keys(citation)
    if passage:
        stored |= {("statute", item[1], passage.lower()) for item in stored if item[0] == "statute"}
    return bool(keys(written)) and all(_found(item, stored) for item in keys(written))


def _found(item: tuple, returned: set[tuple]) -> bool:
    if item[0] == "statute" and item[2] is None:
        return any(other[0] == "statute" and other[1] == item[1] for other in returned)
    return item in returned


def check(answer: str, results: list) -> dict:
    """The citations in the answer: found in the results, not found, and case names that cannot be checked. A plain
    string among the results (the question, a tool's error) counts as given, as number_check counts its figures; text
    inside a tool's result, such as a passage, does not."""
    returned: set[tuple] = set()
    for result in results:
        if isinstance(result, str):
            returned |= keys(result)
        else:
            _returned(result, returned)
    cited = sorted([(position, written, ("statute", section, label))
                    for written, section, label, position in _statutes(answer)]
                   + [(position, written, ("reporter", key)) for written, key, position in _reporters(answer)])
    out: dict[str, list[str]] = {"found": [], "not_found": [], "unchecked": []}
    for _, written, item in cited:
        bucket = out["found" if _found(item, returned) else "not_found"]
        if written not in bucket:
            bucket.append(written)
    for match in _CASE_NAME.finditer(answer):
        if not _FOLLOWED.match(answer, match.end()):
            name = match.group(0).rstrip(",;. ")
            if name not in out["unchecked"]:
                out["unchecked"].append(name)
    return out
