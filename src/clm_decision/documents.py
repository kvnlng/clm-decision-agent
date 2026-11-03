"""Collections of documents an environment can search: statutes, court opinions, policies, manuals.

An environment declares `[documents]` (where the store is, and the embedder) and one `[collections.<name>]` per
collection: what it holds, the metadata it can be filtered on, and the embedding model its passages were loaded
with. The store is PostgreSQL with pgvector (documents_pg.py), loaded by load_documents.py. Searching selects
passages for reading; only the owner's rules decide.
"""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from .scenario import ScenarioError, StepError

BACKENDS = ("postgres",)
FILTER_TYPES = ("string", "integer", "number", "boolean", "date")
_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class Collection:
    """One collection: what the agent reads about it, the fields it can be filtered on, and its embedding model."""
    name: str
    description: str
    filters: dict[str, str]
    embedding_model: str


@dataclass(frozen=True)
class DocumentSettings:  # pylint: disable=too-many-instance-attributes
    """The [documents] table: where PostgreSQL is, and where Ollama embeds."""
    host: str
    port: int
    database: str
    user: str
    password_env: str
    ollama_url: str
    tools: bool = True          # whether the agent gets search_, document_, cites_ and cited_by_ tools


class DocumentStore(Protocol):
    """Where collections are kept. Rows are dicts; see docs/environment.md for their keys."""

    def built(self, collection: str) -> dict | None:
        """When and from what the collection was loaded, its model and counts; None when it holds nothing."""

    def search(self, collection: str, vector: list[float], text: str, filters: dict, limit: int) -> list[dict]:
        """The passages closest to the text, by meaning and by words, among those the filters allow."""

    def document(self, collection: str, id: str, passage: str | None) -> dict:  # pylint: disable=redefined-builtin
        """A document with its passages, or one passage; {} when there is no such document."""

    def cites(self, collection: str, id: str, depth: int) -> list[dict]:  # pylint: disable=redefined-builtin
        """The documents this one cites, and those they cite, to the depth."""

    def cited_by(self, collection: str, id: str, depth: int) -> list[dict]:  # pylint: disable=redefined-builtin
        """The documents that cite this one, and those that cite them, to the depth."""

    def exists(self, collection: str, citation_id: str) -> bool:
        """Whether the collection holds a document with this id."""


class Embedder(Protocol):  # pylint: disable=too-few-public-methods
    """Turns texts into vectors with a named model."""

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """One vector per text."""


def _table(data: dict, key: str, where: str) -> dict:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ScenarioError(f"{where}: must be a table")
    return value


def _text(table: dict, key: str, where: str, default: str | None = None) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ScenarioError(f"{where}.{key}: must be text")
    return value.strip()


def document_settings(data: dict) -> DocumentSettings | None:
    """The [documents] settings, with their defaults; None without the table."""
    if "documents" not in data:
        return None
    table = _table(data, "documents", "documents")
    backend = table.get("backend", "postgres")
    if backend not in BACKENDS:
        raise ScenarioError(f"documents.backend: must be {' or '.join(BACKENDS)}, not {backend!r}")
    port = table.get("port", 5432)
    if not isinstance(port, int) or isinstance(port, bool):
        raise ScenarioError("documents.port: must be a whole number")
    tools = table.get("tools", True)
    if not isinstance(tools, bool):
        raise ScenarioError("documents.tools: must be true or false")
    return DocumentSettings(_text(table, "host", "documents", "127.0.0.1"), port,
                            _text(table, "database", "documents", "clm_documents"),
                            _text(table, "user", "documents", "clm"),
                            _text(table, "password_env", "documents", "CLM_DOCUMENTS_PASSWORD"),
                            _text(table, "ollama_url", "documents", "http://127.0.0.1:11434"), tools)


def collections(data: dict, settings: DocumentSettings | None) -> dict[str, Collection]:
    """The [collections.*] tables, checked; they need a [documents] table."""
    tables = _table(data, "collections", "collections")
    if tables and settings is None:
        raise ScenarioError("collections: need a [documents] table saying where the store is")
    found = {}
    for name, table in tables.items():
        where = f"collections.{name}"
        if not _NAME.fullmatch(name):
            raise ScenarioError(f"{where}: a name is lower case letters, digits and _")
        if not isinstance(table, dict):
            raise ScenarioError(f"{where}: must be a table")
        filters = _table(table, "filters", f"{where}.filters")
        for field, kind in filters.items():
            if not _NAME.fullmatch(field):
                raise ScenarioError(f"{where}.filters.{field}: a name is lower case letters, digits and _")
            if kind not in FILTER_TYPES:
                raise ScenarioError(f"{where}.filters.{field}: type must be {', '.join(FILTER_TYPES)}, not {kind!r}")
        found[name] = Collection(name, _text(table, "description", where), dict(filters),
                                 _text(table, "embedding_model", where))
    return found


def read_settings(path: str | Path) -> tuple[DocumentSettings, dict[str, Collection]]:
    """An environment file's [documents] settings and collections, without loading the rest of it, for the
    loader; raises ScenarioError naming the setting."""
    path = Path(path)
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except OSError as error:
        raise ScenarioError(f"{path}: cannot read: {error.strerror}") from error
    except tomllib.TOMLDecodeError as error:
        raise ScenarioError(f"{path.name}: not valid TOML: {error}") from error
    settings = document_settings(data)
    if settings is None:
        raise ScenarioError(f"{path.name}: has no [documents] table")
    return settings, collections(data, settings)


def documents_password(settings: DocumentSettings) -> str:
    """The password from the variable the settings name; never from the file itself."""
    password = os.environ.get(settings.password_env, "")
    if not password:
        raise ScenarioError(f"documents.password_env: set {settings.password_env} to the database's password")
    return password


def connect(settings: DocumentSettings) -> DocumentStore:
    """The PostgreSQL store; raises ScenarioError in one line when it cannot be reached."""
    from .documents_pg import connect_postgres  # pylint: disable=import-outside-toplevel
    return connect_postgres(settings, documents_password(settings))


class OllamaEmbedder:  # pylint: disable=too-few-public-methods
    """Embeds through Ollama's /api/embed, so text never leaves the machine."""

    def __init__(self, url: str, http=None):
        self.url, self.http = url.rstrip("/"), http

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """One vector per text; raises StepError in one line when Ollama cannot embed them."""
        try:
            if self.http is None:
                with httpx.Client(timeout=600) as http:
                    response = http.post(f"{self.url}/api/embed", json={"model": model, "input": texts})
            else:
                response = self.http.post(f"{self.url}/api/embed", json={"model": model, "input": texts})
            response.raise_for_status()
            vectors = response.json()["embeddings"]
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise StepError(f"embedding with {model} at {self.url}: {_first_line(error)}") from error
        if len(vectors) != len(texts):
            raise StepError(f"embedding with {model}: {len(vectors)} vectors for {len(texts)} texts")
        return vectors


def _first_line(error: Exception) -> str:
    return (str(error).strip().splitlines() or [type(error).__name__])[0]
