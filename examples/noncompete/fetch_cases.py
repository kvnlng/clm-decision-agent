"""Fetch the non-compete opinions of the example's states, and the opinions that cite them, as JSON lines for
clm-load-documents.

    uv run python examples/noncompete/fetch_cases.py fetch     # download and keep candidates (about 11 GB read)
    uv run python examples/noncompete/fetch_cases.py build     # choose the opinions, read their facts, write
    uv run clm-load-documents examples/noncompete/environment.toml cases examples/noncompete/data/cases.jsonl

The opinions come from the Caselaw Access Project (Harvard Law School Library), whose bulk data is free of
restrictions: https://static.case.law. It holds published opinions to about 2020. `fetch` reads one volume at a
time, keeps the opinions of the chosen courts that mention competition or covenants, and deletes the volume. `build`
keeps the opinions about non-competes, plus one step of the opinions in those courts that cite them, and has a local
model read each non-compete opinion's facts. **A fact is kept only if the passage it names contains the words it
quotes;** otherwise it is left empty, never guessed. Nothing leaves the machine but the downloads.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import httpx

from clm_decision.load_documents import split

HERE = Path(__file__).parent
DATA = HERE / "data"
CACHE = DATA / "candidates.jsonl.gz"
OUT = DATA / "cases.jsonl"
SITE = "https://static.case.law"
OLLAMA = "http://127.0.0.1:11434"
MODEL = "gemma4:26b"
CONTEXT = 8192    # tokens: room for the passages and the answer; the model's default is far larger and slower

# Reporter and first volume: published opinions from about 1990. The Caselaw Access Project files a case under its
# official reporter, so Illinois, Massachusetts, Washington and New York need their own.
REPORTERS = {"sw2d": 770, "sw3d": 1, "f3d": 1, "p3d": 1, "nw2d": 450, "ne2d": 550, "ne3d": 1, "so2d": 550,
             "so3d": 1, "ny3d": 1, "ad3d": 1, "cal-app-4th": 1, "cal-app-5th": 1, "cal-4th": 1, "cal-5th": 1,
             "ill-2d": 133, "ill-app-3d": 190, "mass": 406, "mass-app-ct": 28, "wash-2d": 113, "wash-app": 56,
             "ny-2d": 75, "ad2d": 157}
STATES = {"Tex.": "TX", "Cal.": "CA", "Colo.": "CO", "Wash.": "WA", "Ill.": "IL", "Mass.": "MA", "Minn.": "MN",
          "Okla.": "OK", "Fla.": "FL", "N.Y.": "NY", "N.D.": "ND"}
FIFTH_CIRCUIT = "United States Court of Appeals for the Fifth Circuit"
# Loose, to keep every opinion that might cite a non-compete case; then strict, for the non-compete cases
_CANDIDATE = re.compile(r"compet|covenant|restrictive", re.IGNORECASE)
_NON_COMPETE = re.compile(r"covenants? not to compete|non-?competition (?:agreement|covenant|clause|provision)"
                          r"|non-?compete (?:agreement|covenant|clause|provision)s?"
                          r"|(?:post-?employment|employee) (?:restrictive covenant|anti-?competitive agreement)"
                          r"|restrictive covenants? (?:in|of) (?:an|the|his|her|their) employment", re.IGNORECASE)
FACTS = {"role": str, "length_months": (int, float), "area": str, "outcome": ("enforced", "reformed", "void"),
         "reformed_to": str}
PROMPT = """Read these passages of a court opinion about a covenant not to compete. For each fact below that the
passages state, give its value, the label of the passage that states it, and the exact words of that passage that
state it (a short quote, copied character for character). Leave out any fact the passages do not state.

- role: the restrained person's job, such as "sales manager"
- length_months: how long the restriction lasts, in months, as a number
- area: where it applies, such as "nationwide" or "50 miles of Houston"
- outcome: what the court did with the covenant: "enforced", "reformed" or "void"
- reformed_to: if reformed, the narrower terms the court allowed

Answer with JSON only: {"role": {"value": ..., "passage": "...", "quote": "..."}, ...}

PASSAGES:
"""


def opinion_text(case: dict) -> str:
    """The text of every opinion in the case, majority first."""
    return "\n\n".join(opinion.get("text", "") for opinion in case["casebody"].get("opinions", []))


def is_non_compete(text: str) -> bool:
    """Whether an opinion is about a covenant not to compete, not a deed's restrictive covenant."""
    return bool(_NON_COMPETE.search(text))


def state_of(case: dict) -> str | None:
    """The state whose law the opinion applies: its court's, or for the Fifth Circuit, Texas when it says so."""
    if case["court"]["name"] == FIFTH_CIRCUIT:
        return "TX" if re.search(r"\bTexas law\b", opinion_text(case)) else None
    return STATES.get(case["jurisdiction"]["name"])


def court_level(case: dict) -> str:
    """supreme, appeals or federal: what weight the court's ruling may carry is the owner's rule, not this."""
    name = case["court"]["name"]
    if name.startswith("United States"):
        return "federal"
    # New York's Supreme Court sits below its Court of Appeals, as it sat below the Court for the Correction of Errors
    # before 1847: a name the source also gives some later Court of Appeals opinions.
    if "New York" in name:
        return "supreme" if name in ("New York Court of Appeals", "New York Court for the Correction of Errors") \
            else "appeals"
    return "supreme" if "Supreme" in name else "appeals"


def checked_facts(facts: dict, passages: list[dict]) -> dict:
    """The facts whose named passage contains their quoted words, and whose value is of the right kind."""
    texts = {passage["label"]: " ".join(passage["text"].split()).lower() for passage in passages}
    kept = {}
    for name, fact in facts.items():
        kind = FACTS.get(name)
        if kind is None or not isinstance(fact, dict):
            continue
        value, label, quote = fact.get("value"), str(fact.get("passage")).strip("[] "), fact.get("quote")
        if isinstance(kind, tuple) and all(isinstance(item, str) for item in kind):
            right = value in kind
        else:
            right = isinstance(value, kind) and not isinstance(value, bool) and value not in ("", None)
        if right and isinstance(quote, str) and quote.strip() and \
                " ".join(quote.split()).lower() in texts.get(label, ""):
            kept[name] = {"value": value, "passage": label, "quote": quote}
    return kept


def _citing_passage(passages: list[dict], cite: str) -> str | None:
    return next((passage["label"] for passage in passages if cite in passage["text"]), None)


def document(case: dict, facts: dict) -> dict:
    """The opinion as a JSON-lines document: passages, metadata with its checked facts, and what it cites."""
    passages = split(opinion_text(case))
    citation = next((c["cite"] for c in case["citations"] if c["type"] == "official"), case["citations"][0]["cite"])
    decided = case["decision_date"]
    metadata = {"state": state_of(case), "court": case["court"]["name_abbreviation"], "court_level": court_level(case),
                "year": int(decided[:4])}
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", decided):
        metadata["decided"] = decided
    kept = checked_facts(facts, passages)
    metadata |= {name: fact["value"] for name, fact in kept.items()}
    if kept:
        metadata["sources"] = {name: fact["passage"] for name, fact in kept.items()}
    cites = [{"id": f"cap-{target}", "passage": _citing_passage(passages, item["cite"])}
             for item in case.get("cites_to", []) for target in item.get("case_ids", [])[:1]]
    return {"id": f"cap-{case['id']}", "citation": citation, "title": case["name_abbreviation"],
            "metadata": metadata, "passages": passages, "cites": cites}


def _volumes(http: httpx.Client, reporter: str) -> list[int]:
    listing = http.get(f"{SITE}/{reporter}/").text
    return sorted({int(n) for n in re.findall(rf"{SITE}/{re.escape(reporter)}/(\d+)/'", listing)})


def fetch() -> None:
    """Read every chosen volume once and keep the candidate opinions of the chosen courts."""
    DATA.mkdir(exist_ok=True)
    done_path = DATA / "volumes-done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    with httpx.Client(timeout=120, follow_redirects=True) as http, gzip.open(CACHE, "at", encoding="utf-8") as out, \
            open(done_path, "a", encoding="utf-8") as log:
        for reporter, first in REPORTERS.items():
            for volume in (v for v in _volumes(http, reporter) if v >= first):
                name = f"{reporter}/{volume}"
                if name in done:
                    continue
                response = http.get(f"{SITE}/{name}.zip")
                response.raise_for_status()
                kept = 0
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    for member in archive.namelist():
                        if not re.search(r"/?json/[^/]+\.json$", member):
                            continue
                        case = json.loads(archive.read(member))
                        text = opinion_text(case)
                        if (case["court"]["name"] == FIFTH_CIRCUIT or case["jurisdiction"]["name"] in STATES) \
                                and _CANDIDATE.search(text):
                            out.write(json.dumps(case) + "\n")
                            kept += 1
                log.write(name + "\n")
                log.flush()
                print(f"{name}: {kept} kept", flush=True)


def _relevant(passages: list[dict], limit: int = 8000) -> str:
    """The passages most likely to state the facts, labelled, up to `limit` characters."""
    chosen = [p for p in passages if re.search(r"compet|covenant|month|year|mile|reform|enforc", p["text"], re.I)]
    out, size = [], 0
    for passage in chosen:
        block = f"[{passage['label']}] {passage['text']}"
        if size + len(block) > limit:
            break
        out.append(block)
        size += len(block)
    return "\n\n".join(out)


def read_facts(http: httpx.Client, case: dict, model: str) -> dict:
    """The facts a local model reads in the opinion; {} when it gives no usable answer."""
    passages = split(opinion_text(case))
    response = http.post(f"{OLLAMA}/api/generate", json={"model": model, "prompt": PROMPT + _relevant(passages),
                                                         "format": "json", "stream": False, "think": False,
                                                         "options": {"temperature": 0, "num_ctx": CONTEXT}})
    try:
        facts = json.loads(response.json()["response"])
    except (KeyError, ValueError):
        return {}
    return facts if isinstance(facts, dict) else {}


def build(model: str, states: set[str] | None) -> None:
    """Choose the non-compete opinions and one step of their citers; read the facts; write cases.jsonl."""
    cases = {}
    with gzip.open(CACHE, "rt", encoding="utf-8") as file:
        for line in file:
            case = json.loads(line)
            if state_of(case):
                cases[case["id"]] = case
    core = {ident for ident, case in cases.items() if is_non_compete(opinion_text(case))}
    citers = {ident for ident, case in cases.items() if ident not in core and any(
        target in core for item in case.get("cites_to", []) for target in item.get("case_ids", []))}
    print(f"{len(core)} non-compete opinions, {len(citers)} citing them", flush=True)
    facts_path = DATA / f"facts-{model.replace(':', '-')}.jsonl"
    read = {}
    if facts_path.exists():
        read = {item["id"]: item["facts"] for item in map(json.loads, facts_path.read_text().splitlines())}
    with httpx.Client(timeout=600) as http, open(facts_path, "a", encoding="utf-8") as log:
        for n, ident in enumerate(sorted(core), 1):
            if ident in read or (states and state_of(cases[ident]) not in states):
                continue
            read[ident] = read_facts(http, cases[ident], model)
            log.write(json.dumps({"id": ident, "facts": read[ident]}) + "\n")
            log.flush()
            if n % 25 == 0:
                print(f"facts read for {n} of {len(core)}", flush=True)
    with open(OUT, "w", encoding="utf-8") as out:
        for ident in sorted(core | citers):
            out.write(json.dumps(document(cases[ident], read.get(ident, {})), ensure_ascii=False) + "\n")
    print(f"wrote {len(core | citers)} documents to {OUT}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Fetch the example's court opinions.")
    parser.add_argument("step", choices=("fetch", "build"))
    parser.add_argument("--model", default=MODEL, help=f"the local model that reads facts (default {MODEL})")
    parser.add_argument("--states", help="read facts only for these states, such as TX,CO")
    args = parser.parse_args(argv)
    if args.step == "fetch":
        fetch()
    else:
        build(args.model, set(args.states.split(",")) if args.states else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
