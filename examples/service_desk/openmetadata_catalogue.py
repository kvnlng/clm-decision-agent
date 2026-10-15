"""Reads tables from OpenMetadata's REST API and turns them into catalogue entries for the framework.

OpenMetadata follows its graph of tables, owners, tags and lineage itself; this module only asks
for the results and writes each table as an entry: a name, a description and facts in sentences.

The request paths and field names below are written from OpenMetadata's documentation and have
been run only against mock_openmetadata.py. Check them against your instance and its version
(for example, older versions return `owner` rather than `owners`).
"""
from __future__ import annotations

import re
from collections import Counter
from urllib.parse import quote

import httpx
from langchain_core.tools import ToolException

TAG_SENTENCES = {"PII.Sensitive": "{name} contains sensitive personal data.",
                 "PII.NonSensitive": "{name} contains personal data that is not sensitive."}
TIER = re.compile(r"Tier\.Tier(\d+)")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MARKS = re.compile(r"[*_`#>]+")


def plain(text: str | None) -> str:
    """Markdown as one line of plain text: links become their words, paragraphs become sentences."""
    text = MARKS.sub("", LINK.sub(r"\1", text or ""))
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", text) if part.strip()]
    # A paragraph that is a heading or a fragment gets a full stop, so the next one reads as a new sentence
    ended = [p if p.endswith((".", "!", "?")) else p + "." for p in paragraphs[:-1]]
    return " ".join(ended + paragraphs[-1:])


def _tag_fact(name: str, tag: str) -> str:
    tier = TIER.fullmatch(tag)
    if tier:
        return f"{name} is a tier {tier.group(1)} dataset."
    return TAG_SENTENCES.get(tag, "{name} is tagged " + tag + ".").format(name=name)


def upstream(response: dict) -> list[str]:
    """The names of the tables a lineage response says its table is built from."""
    target = (response.get("entity") or {}).get("id")
    names = {node["id"]: node.get("name", "") for node in response.get("nodes") or []}
    return [names[edge["fromEntity"]] for edge in response.get("upstreamEdges") or []
            if edge.get("toEntity") == target and edge.get("fromEntity") in names]


def to_entries(tables: list[dict], lineage: dict[str, dict]) -> list[dict]:
    """OpenMetadata tables and their lineage (by fully qualified name) -> catalogue entries.
    Tables without a description are left out: CLM chooses by description. A name used by
    tables in two schemas is replaced by each one's fully qualified name, so entries stay distinct."""
    counts = Counter(table["name"] for table in tables)
    entries = []
    for table in tables:
        name, description = table["name"], plain(table.get("description"))
        if counts[name] > 1:
            name = table.get("fullyQualifiedName") or name
        if not description:
            continue
        facts = [f"{owner.get('displayName') or owner['name']} owns {name}." for owner in table.get("owners") or []]
        facts += [_tag_fact(name, tag["tagFQN"]) for tag in table.get("tags") or []]
        sources = upstream(lineage.get(table.get("fullyQualifiedName")) or {})
        facts += [f"{name} is built from {source}." for source in sources]
        entries.append({"name": name, "description": description, "facts": facts})
    return entries


def _get(base: str, path: str, token: str | None, params: dict) -> dict:
    """One GET to the API; any failure is a ToolException naming the address."""
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = base.rstrip("/") + path
    try:
        response = httpx.get(url, params=params, headers=headers, timeout=10)
    except httpx.HTTPError as error:
        raise ToolException(f"OpenMetadata at {base} could not be reached: {error}") from error
    if response.status_code >= 400:
        raise ToolException(f"OpenMetadata answered {response.status_code} for {url}")
    return response.json()


def fetch(url: str, token: str | None, domain: str | None = None, page: int = 100) -> tuple[list[dict], dict]:
    """Tables with their owners, tags and domains, and each one's upstream lineage:
    -> (tables, lineage by fully qualified name).
    `domain` keeps the tables of one domain; it is applied here because the server's filter varies by version."""
    tables, after = [], None
    while True:
        params = {"fields": "owners,tags,domains", "limit": page, **({"after": after} if after else {})}
        answer = _get(url, "/api/v1/tables", token, params)
        tables += answer.get("data") or []
        after = (answer.get("paging") or {}).get("after")
        if not after:
            break
    if domain:
        tables = [table for table in tables if domain in [d.get("name") for d in table.get("domains") or []]]
    lineage = {}
    for table in tables:
        fqn = table["fullyQualifiedName"]
        lineage[fqn] = _get(url, "/api/v1/lineage/table/name/" + quote(fqn, safe=""), token,
                            {"upstreamDepth": 1, "downstreamDepth": 0})
    return tables, lineage
