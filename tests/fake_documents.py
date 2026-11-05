"""A document store and an embedder without PostgreSQL or Ollama, for the tests.

FakeStore keeps documents in memory and ranks passages by the words they share with the text; FakeEmbedder gives a
fixed-length vector of letter counts. Both follow the protocols in clm_decision.documents.
"""
# Fakes mirror real interfaces, so some arguments go unused.
# pylint: disable=missing-function-docstring,unused-argument
from __future__ import annotations


def _matches(value, wanted) -> bool:
    if isinstance(wanted, list):
        return value in wanted
    if isinstance(wanted, dict):
        return value is not None and wanted.get("from", value) <= value <= wanted.get("to", value)
    return value == wanted


class FakeStore:
    """Documents by collection; `built` as clm-load-documents records it."""

    def __init__(self):
        self.collections: dict[str, list[dict]] = {}
        self.info: dict[str, dict] = {}

    def replace(self, collection: str, documents: list[dict], vectors: list[list[list[float]]], built: dict) -> None:
        self.collections[collection] = [dict(document) for document in documents]
        self.info[collection] = dict(built)

    def built(self, collection: str) -> dict | None:
        return self.info.get(collection) if self.collections.get(collection) else None

    def _documents(self, collection: str) -> dict[str, dict]:
        return {document["id"]: document for document in self.collections.get(collection, [])}

    def search(self, collection: str, vector, text: str, filters: dict, limit: int) -> list[dict]:
        words = set(text.lower().split())
        rows = []
        for document in self.collections.get(collection, []):
            if not all(_matches(document["metadata"].get(field), wanted) for field, wanted in filters.items()):
                continue
            for passage in document["passages"]:
                score = len(words & set(passage["text"].lower().split()))
                if score:
                    rows.append({"document": document["id"], "citation": document["citation"],
                                 "title": document["title"], "metadata": document["metadata"],
                                 "passage": passage["label"], "text": passage["text"], "score": float(score)})
        return sorted(rows, key=lambda row: -row["score"])[:limit]

    def document(self, collection: str, id: str, passage: str | None) -> dict:  # pylint: disable=redefined-builtin
        document = self._documents(collection).get(id)
        if document is None:
            return {}
        passages = [p for p in document["passages"] if passage is None or p["label"] == passage]
        return {key: document[key] for key in ("id", "citation", "title", "metadata")} | {"passages": passages}

    def _walk(self, collection: str, id: str, depth: int, forward: bool) -> list[dict]:  # pylint: disable=redefined-builtin,too-many-locals
        documents = self._documents(collection)
        links = [(source["id"], cite["id"] if isinstance(cite, dict) else cite,
                  cite.get("passage") if isinstance(cite, dict) else None)
                 for source in documents.values() for cite in source.get("cites", [])]
        found, frontier = [], [id]
        seen = {id}
        for step in range(1, depth + 1):
            following = []
            for current in frontier:
                for source, target, passage in links:
                    near, far = (source, target) if forward else (target, source)
                    if near == current and far not in seen:
                        seen.add(far)
                        following.append(far)
                        citation = documents.get(far, {}).get("citation")
                        found.append({"document": far, "citation": citation, "depth": step,
                                      "passage": None if forward else passage})
            frontier = following
        return found

    def cites(self, collection: str, id: str, depth: int) -> list[dict]:  # pylint: disable=redefined-builtin
        return self._walk(collection, id, depth, True)

    def cited_by(self, collection: str, id: str, depth: int) -> list[dict]:  # pylint: disable=redefined-builtin
        return self._walk(collection, id, depth, False)

    def exists(self, collection: str, citation_id: str) -> bool:
        return citation_id in self._documents(collection)

    def ids(self, collection: str, limit: int) -> list[str]:
        return sorted(self._documents(collection))[:limit]


class FakeEmbedder:
    """Letter counts as the vector; records the models and texts it was asked for."""
    dimensions = 26

    def __init__(self):
        self.asked: list[tuple[str, list[str]]] = []

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self.asked.append((model, list(texts)))
        return [[float(text.lower().count(chr(ord("a") + i))) for i in range(self.dimensions)] for text in texts]
