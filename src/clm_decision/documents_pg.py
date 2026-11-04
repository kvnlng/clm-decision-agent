"""Collections in PostgreSQL: passages found by meaning (pgvector) and by words (full-text search), and citations.

The driver is the optional `documents` extra (psycopg and pgvector), imported only when an environment has a
[documents] table. Four tables hold every collection: `collections` (when and from what each was built), `documents`,
`passages` (text, full-text vector and embedding) and `citations` (which passage of which document cites which
document). load_documents.py fills them.
"""
from __future__ import annotations

import json

from .scenario import ScenarioError, StepError

# How far down each ranking a passage still counts, and the constant of reciprocal rank fusion
CANDIDATES = 50
FUSION_K = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    name text PRIMARY KEY, built_at text NOT NULL, source text NOT NULL, model text NOT NULL,
    dimensions integer NOT NULL, documents integer NOT NULL, passages integer NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    collection text NOT NULL, id text NOT NULL, citation text NOT NULL, title text NOT NULL,
    metadata jsonb NOT NULL, PRIMARY KEY (collection, id));
CREATE TABLE IF NOT EXISTS passages (
    collection text NOT NULL, document text NOT NULL, n integer NOT NULL, label text NOT NULL, text text NOT NULL,
    tsv tsvector NOT NULL, embedding vector NOT NULL, PRIMARY KEY (collection, document, n));
CREATE TABLE IF NOT EXISTS citations (
    collection text NOT NULL, source text NOT NULL, source_passage text, target text NOT NULL);
CREATE INDEX IF NOT EXISTS passages_tsv ON passages USING gin (tsv);
CREATE INDEX IF NOT EXISTS citations_source ON citations (collection, source);
CREATE INDEX IF NOT EXISTS citations_target ON citations (collection, target);
"""

_NO_DRIVER = "documents: needs the driver: install the documents extra, pip install 'clm-decision-agent[documents]'"


def _first_line(error: Exception) -> str:
    return (str(error).strip().splitlines() or [type(error).__name__])[0]


def connect_postgres(settings, password: str) -> PostgresStore:
    """A store that has reached the database, with the vector extension and the tables; raises ScenarioError in one
    line when it cannot."""
    try:
        import psycopg  # pylint: disable=import-outside-toplevel
        from pgvector.psycopg import register_vector  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise ScenarioError(_NO_DRIVER) from error
    where = f"{settings.host}:{settings.port}/{settings.database}"
    try:
        connection = psycopg.connect(host=settings.host, port=settings.port, dbname=settings.database,
                                     user=settings.user, password=password, connect_timeout=5, autocommit=True)
    except psycopg.Error as error:
        raise ScenarioError(f"documents: cannot connect to {where}: {_first_line(error)}") from error
    try:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except psycopg.Error as error:
        connection.close()
        raise ScenarioError(f"documents: {where} has no vector extension (brew install pgvector): "
                            f"{_first_line(error)}") from error
    register_vector(connection)
    connection.execute(SCHEMA)
    return PostgresStore(connection, where)


def _filter_sql(filters: dict) -> tuple[str, list]:
    """SQL conditions on documents' metadata (alias d): a value, a list of values, or {"from", "to"}."""
    clauses, values = [], []
    for field, wanted in filters.items():
        if isinstance(wanted, list):
            clauses.append("d.metadata -> %s = ANY(%s::jsonb[])")
            values += [field, [json.dumps(value) for value in wanted]]
        elif isinstance(wanted, dict):
            for key, op in (("from", ">="), ("to", "<=")):
                if key in wanted:
                    numeric = isinstance(wanted[key], (int, float)) and not isinstance(wanted[key], bool)
                    cast = "(d.metadata ->> %s)::numeric" if numeric else "d.metadata ->> %s"
                    clauses.append(f"{cast} {op} %s")
                    values += [field, wanted[key]]
        else:
            clauses.append("d.metadata -> %s = %s::jsonb")
            values += [field, json.dumps(wanted)]
    return "".join(f" AND {clause}" for clause in clauses), values


class PostgresStore:
    """A DocumentStore on one PostgreSQL connection."""

    def __init__(self, connection, where: str):
        self.connection, self.where = connection, where

    def _rows(self, sql: str, params) -> list[tuple]:
        import psycopg  # pylint: disable=import-outside-toplevel
        try:
            return self.connection.execute(sql, params).fetchall()
        except psycopg.Error as error:
            raise StepError(f"documents at {self.where}: {_first_line(error)}") from error

    def replace(self, collection: str, documents: list[dict], vectors: list[list[list[float]]], built: dict) -> None:
        """Replace the collection with these documents, one vector per passage, in one transaction."""
        import numpy  # pylint: disable=import-outside-toplevel
        with self.connection.transaction():
            for table, column in (("citations", "collection"), ("passages", "collection"),
                                  ("documents", "collection"), ("collections", "name")):
                self.connection.execute(f"DELETE FROM {table} WHERE {column} = %s", (collection,))
            with self.connection.cursor() as cursor:
                for document, document_vectors in zip(documents, vectors):
                    cursor.execute("INSERT INTO documents VALUES (%s, %s, %s, %s, %s)",
                                   (collection, document["id"], document["citation"], document["title"],
                                    json.dumps(document["metadata"])))
                    cursor.executemany(
                        "INSERT INTO passages VALUES (%s, %s, %s, %s, %s, to_tsvector('english', %s), %s)",
                        [(collection, document["id"], n, passage["label"], passage["text"], passage["text"],
                          numpy.array(vector, dtype=numpy.float32))
                         for n, (passage, vector) in enumerate(zip(document["passages"], document_vectors))])
                    cursor.executemany("INSERT INTO citations VALUES (%s, %s, %s, %s)",
                                       [(collection, document["id"], cite.get("passage"), cite["id"])
                                        for cite in document.get("cites", [])])
            self.connection.execute(
                "INSERT INTO collections VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (collection, built["built_at"], built["source"], built["model"], built["dimensions"],
                 len(documents), sum(len(document["passages"]) for document in documents)))

    def built(self, collection: str) -> dict | None:
        """When and from what the collection was built, with its counts; None when it holds nothing."""
        rows = self._rows("SELECT built_at, source, model, dimensions, documents, passages FROM collections "
                          "WHERE name = %s", (collection,))
        if not rows or not rows[0][4]:
            return None
        keys = ("built_at", "source", "model", "dimensions", "documents", "passages")
        return dict(zip(keys, rows[0]))

    def search(self, collection: str, vector: list[float], text: str, filters: dict, limit: int) -> list[dict]:
        """Reciprocal rank fusion of the nearest passages by cosine distance and the best by ts_rank."""
        import numpy  # pylint: disable=import-outside-toplevel
        where, values = _filter_sql(filters)
        embedding = numpy.array(vector, dtype=numpy.float32)
        sql = f"""
            WITH meaning AS (
                SELECT p.document, p.n, row_number() OVER (ORDER BY p.embedding <=> %s) AS rank
                FROM passages p JOIN documents d ON d.collection = p.collection AND d.id = p.document
                WHERE p.collection = %s{where} ORDER BY p.embedding <=> %s LIMIT {CANDIDATES}),
            words AS (
                SELECT p.document, p.n, row_number() OVER (ORDER BY ts_rank(p.tsv, q) DESC) AS rank
                FROM passages p JOIN documents d ON d.collection = p.collection AND d.id = p.document,
                     websearch_to_tsquery('english', %s) q
                WHERE p.collection = %s AND p.tsv @@ q{where} ORDER BY ts_rank(p.tsv, q) DESC LIMIT {CANDIDATES}),
            fused AS (
                SELECT document, n, sum(1.0 / ({FUSION_K} + rank)) AS score
                FROM (SELECT * FROM meaning UNION ALL SELECT * FROM words) ranked GROUP BY document, n)
            SELECT d.id, d.citation, d.title, d.metadata, p.label, p.text, f.score
            FROM fused f JOIN passages p ON p.collection = %s AND p.document = f.document AND p.n = f.n
                 JOIN documents d ON d.collection = p.collection AND d.id = p.document
            ORDER BY f.score DESC, d.id, p.n LIMIT %s"""
        params = [embedding, collection, *values, embedding, text, collection, *values, collection, limit]
        keys = ("document", "citation", "title", "metadata", "passage", "text", "score")
        return [dict(zip(keys, row)) | {"score": float(row[6])} for row in self._rows(sql, params)]

    def document(self, collection: str, id: str, passage: str | None) -> dict:  # pylint: disable=redefined-builtin
        """The document with its passages, or only the passage labelled `passage`; {} when there is none."""
        rows = self._rows("SELECT id, citation, title, metadata FROM documents WHERE collection = %s AND id = %s",
                          (collection, id))
        if not rows:
            return {}
        passages = self._rows("SELECT label, text FROM passages WHERE collection = %s AND document = %s"
                              " AND (%s::text IS NULL OR label = %s) ORDER BY n", (collection, id, passage, passage))
        return dict(zip(("id", "citation", "title", "metadata"), rows[0])) | {
            "passages": [{"label": label, "text": text} for label, text in passages]}

    def _walk(self, collection: str, id: str, depth: int, forward: bool) -> list[dict]:  # pylint: disable=redefined-builtin
        """Breadth first, one query a step, each document once at its nearest depth: a walk of paths would grow with
        every path between two documents."""
        near, far = ("source", "target") if forward else ("target", "source")
        sql = f"""
            SELECT DISTINCT ON (c.{far}) c.{far}, d.citation, c.source_passage
            FROM citations c LEFT JOIN documents d ON d.collection = c.collection AND d.id = c.{far}
            WHERE c.collection = %s AND c.{near} = ANY(%s) AND NOT c.{far} = ANY(%s)
            ORDER BY c.{far}, c.source_passage"""
        found, frontier, seen = [], [id], [id]
        for step in range(1, depth + 1):
            rows = self._rows(sql, (collection, frontier, seen)) if frontier else []
            found += [{"document": document, "citation": citation, "depth": step, "passage": passage}
                      for document, citation, passage in rows]
            frontier = [row[0] for row in rows]
            seen += frontier
        return found

    def cites(self, collection: str, id: str, depth: int) -> list[dict]:  # pylint: disable=redefined-builtin
        """The documents this one cites, to the depth, nearest first."""
        return self._walk(collection, id, depth, True)

    def cited_by(self, collection: str, id: str, depth: int) -> list[dict]:  # pylint: disable=redefined-builtin
        """The documents citing this one, to the depth, nearest first, with the citing passage."""
        return self._walk(collection, id, depth, False)

    def exists(self, collection: str, citation_id: str) -> bool:
        """Whether the collection holds a document with this id."""
        return bool(self._rows("SELECT 1 FROM documents WHERE collection = %s AND id = %s",
                               (collection, citation_id)))
