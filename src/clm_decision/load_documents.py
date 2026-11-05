"""Load a collection of documents from JSON lines, replacing what the collection held.

    set -a; . ./.postgres-local.env; set +a
    uv run clm-load-documents ENVIRONMENT COLLECTION FILE.jsonl [--source TEXT]

Each line is one document: `id`, `citation` and `title` (text), `metadata` (only the collection's declared filters,
plus `sources` saying which passage each value came from), and either `passages` (a list of {"label", "text"}, such as
a statute's subsections) or `text`, which is split into passages here. `cites` lists the ids it cites, as text or as
{"id", "passage"} naming the citing passage; a cited id outside the file is kept and counted. Each passage is
embedded with the collection's model, through Ollama, so no text leaves the machine. The collection records when it
was built and from what.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from .documents import Collection, Embedder, connect, read_settings
from .scenario import ScenarioError, StepError

PASSAGE_CHARS = 1500
BATCH = 32
_TYPES = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,), "date": (str,)}


def split(text: str) -> list[dict]:
    """Paragraphs as passages, each cut at a space to at most PASSAGE_CHARS characters; labelled 1, 2, ..."""
    pieces = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        while len(paragraph) > PASSAGE_CHARS:
            cut = paragraph.rfind(" ", 0, PASSAGE_CHARS + 1)
            cut = cut if cut > 0 else PASSAGE_CHARS
            pieces.append(paragraph[:cut])
            paragraph = paragraph[cut:].strip()
        if paragraph:
            pieces.append(paragraph)
    return [{"label": str(n), "text": piece} for n, piece in enumerate(pieces, 1)]


def _value_ok(kind: str, value) -> bool:
    if isinstance(value, bool) and kind != "boolean":
        return False
    if not isinstance(value, _TYPES[kind]):
        return False
    if kind == "date":
        # The form search filters and in_force compare as text; fromisoformat alone also takes 20240115
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return False
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
    return True


def _metadata(metadata, collection: Collection, where: str) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError(f"{where}: metadata must be an object")
    for field, value in metadata.items():
        if field == "sources":
            if not isinstance(value, dict):
                raise ValueError(f"{where}: metadata.sources must be an object")
            continue
        if field not in collection.filters:
            raise ValueError(f"{where}: metadata.{field} is not a filter of {collection.name} "
                             f"({', '.join(sorted(collection.filters))})")
        kind = collection.filters[field]
        if value is not None and not _value_ok(kind, value):
            raise ValueError(f"{where}: metadata.{field} must be {'a YYYY-MM-DD date' if kind == 'date' else kind}, "
                             f"not {value!r}")
    return metadata


def _passages(line: dict, where: str) -> list[dict]:
    if "passages" in line:
        passages = line["passages"]
        if not isinstance(passages, list) or not passages or not all(
                isinstance(p, dict) and isinstance(p.get("label"), str) and isinstance(p.get("text"), str)
                and p["text"].strip() for p in passages):
            raise ValueError(f"{where}: passages must be a list of {{\"label\", \"text\"}} with text")
        return [{"label": p["label"], "text": p["text"].strip()} for p in passages]
    if not isinstance(line.get("text"), str) or not line["text"].strip():
        raise ValueError(f"{where}: needs passages, or text")
    return split(line["text"])


def _cites(cites, where: str) -> list[dict]:
    if not isinstance(cites, list):
        raise ValueError(f"{where}: cites must be a list")
    found = []
    for cite in cites:
        if isinstance(cite, str):
            found.append({"id": cite, "passage": None})
        elif isinstance(cite, dict) and isinstance(cite.get("id"), str) \
                and isinstance(cite.get("passage"), (str, type(None))):
            found.append({"id": cite["id"], "passage": cite.get("passage")})
        else:
            raise ValueError(f"{where}: each of cites is an id, or {{\"id\", \"passage\"}}")
    return found


def _nul(value, name: str) -> str | None:
    """The name of the first text holding a NUL character, which PostgreSQL text cannot store; None when none does."""
    if isinstance(value, str):
        return name if "\x00" in value else None
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, item in items:
        if "\x00" in str(key):
            return f"{name}.{key}"
        found = _nul(item, f"{name}.{key}" if isinstance(key, str) else f"{name}[{key}]")
        if found:
            return found
    return None


def read_jsonl(path: str | Path, collection: Collection) -> list[dict]:
    """The file's documents, checked; raises ValueError naming the file and line."""
    path = Path(path)
    documents, lines = [], {}
    with open(path, encoding="utf-8-sig") as file:
        for number, text in enumerate(file, 1):
            if not text.strip():
                continue
            where = f"{path.name}:{number}"
            try:
                line = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"{where}: not valid JSON: {error.msg}") from error
            if not isinstance(line, dict):
                raise ValueError(f"{where}: must be an object")
            for key in ("id", "citation", "title"):
                if not isinstance(line.get(key), str) or not line[key].strip():
                    raise ValueError(f"{where}: needs {key}, as text")
            if line["id"] in lines:
                raise ValueError(f"{where}: id {line['id']} is already on line {lines[line['id']]}")
            lines[line["id"]] = number
            document = {"id": line["id"], "citation": line["citation"], "title": line["title"],
                        "metadata": _metadata(line.get("metadata", {}), collection, where),
                        "passages": _passages(line, where), "cites": _cites(line.get("cites", []), where)}
            nul = next(filter(None, (_nul(value, key) for key, value in document.items())), None)
            if nul:
                raise ValueError(f"{where}: {nul} contains a NUL character")
            documents.append(document)
    return documents


def load(store, embedder: Embedder, collection: Collection, documents: list[dict], *,  # pylint: disable=too-many-locals,too-many-arguments
         source: str, now: str | None = None) -> dict:
    """Embed every passage and replace the collection -> the counts and when it was built."""
    texts = [passage["text"] for document in documents for passage in document["passages"]]
    flat = []
    for start in range(0, len(texts), BATCH):
        flat += embedder.embed(collection.embedding_model, texts[start:start + BATCH])
    sizes = {len(vector) for vector in flat}
    if len(sizes) > 1:
        raise StepError(f"embedding with {collection.embedding_model}: vectors of different sizes {sorted(sizes)}")
    vectors, at = [], 0
    for document in documents:
        vectors.append(flat[at:at + len(document["passages"])])
        at += len(document["passages"])
    built_at = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    store.replace(collection.name, documents, vectors, {"built_at": built_at, "source": source,
                                                        "model": collection.embedding_model,
                                                        "dimensions": sizes.pop() if sizes else 0})
    ids = {document["id"] for document in documents}
    cites = [cite for document in documents for cite in document["cites"]]
    return {"documents": len(documents), "passages": len(texts), "citations": len(cites),
            "outside": sum(cite["id"] not in ids for cite in cites), "built_at": built_at}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Load a collection of documents from JSON lines, replacing it.")
    parser.add_argument("environment", help="path to the environment's environment.toml")
    parser.add_argument("collection", help="the collection's name, from [collections.<name>]")
    parser.add_argument("file", help="the documents, one JSON object per line")
    parser.add_argument("--source", help="where the documents came from (default: the file's name)")
    args = parser.parse_args(argv)
    try:
        settings, collections = read_settings(args.environment)
        if args.collection not in collections:
            raise ScenarioError(f"no collection {args.collection} in {Path(args.environment).name} "
                                f"({', '.join(sorted(collections)) or 'none declared'})")
        collection = collections[args.collection]
        documents = read_jsonl(args.file, collection)
        from .documents import OllamaEmbedder  # pylint: disable=import-outside-toplevel
        store = connect(settings)
        counts = load(store, OllamaEmbedder(settings.ollama_url, timeout=600), collection, documents,
                      source=args.source or Path(args.file).name)
    except (ScenarioError, StepError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Loaded {counts['documents']} documents, {counts['passages']} passages and {counts['citations']} "
          f"citations ({counts['outside']} to documents outside the collection) into {collection.name}, "
          f"built {counts['built_at']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
