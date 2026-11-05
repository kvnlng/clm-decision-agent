"""Tests for documents_pg.py and the Ollama embedder: collections in PostgreSQL with pgvector.

Those marked postgres need a local PostgreSQL with pgvector, a role `clm` and a database `clm_documents_test` that
can be wiped, with the password in CLM_TEST_POSTGRES_PASSWORD: `uv run pytest -m postgres tests/test_documents_pg.py`.
"""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror psycopg's interface, so some arguments go unused.
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument
import contextlib
import json
import os

import httpx
import pgvector.psycopg
import psycopg
import pytest
from fake_documents import FakeEmbedder

from clm_decision import documents, documents_pg
from clm_decision.documents import DocumentSettings, OllamaEmbedder
from clm_decision.scenario import ScenarioError, StepError

TEST_DB = DocumentSettings("127.0.0.1", 5432, "clm_documents_test", "clm", "CLM_TEST_POSTGRES_PASSWORD",
                           "http://127.0.0.1:11434")
NEVER = DocumentSettings("127.0.0.1", 1, "nowhere", "clm", "X", "http://127.0.0.1:11434")

DOCUMENTS = [
    {"id": "tx-15.50", "citation": "Tex. Bus. & Com. Code § 15.50", "title": "Criteria for enforceability",
     "metadata": {"state": "TX", "year": 1989},
     "passages": [{"label": "(a)", "text": "A covenant not to compete is enforceable if it contains reasonable "
                                           "limitations as to time, geographical area, and scope of activity."}],
     "cites": []},
    {"id": "case-a", "citation": "100 S.W.3d 1", "title": "Case A", "metadata": {"state": "TX", "year": 2019},
     "passages": [{"label": "1", "text": "The sales manager was barred nationwide for two years."},
                  {"label": "2", "text": "We reform the covenant to the territory he served."}],
     "cites": [{"id": "tx-15.50", "passage": "2"}]},
    {"id": "case-b", "citation": "200 S.W.3d 5", "title": "Case B", "metadata": {"state": "TX", "year": 2022},
     "passages": [{"label": "1", "text": "Following Case A, the restriction on the account executive is narrowed."}],
     "cites": [{"id": "case-a", "passage": "1"}]},
    {"id": "ny-1", "citation": "93 N.Y.2d 382", "title": "New York case", "metadata": {"state": "NY", "year": 1999},
     "passages": [{"label": "1", "text": "An accountant's restriction is enforced only to protect legitimate "
                                         "interests."}], "cites": []},
]


def built():
    return {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "toy-embed", "dimensions": 26}


def vectors(documents=None):
    embedder = FakeEmbedder()
    return [embedder.embed("toy-embed", [p["text"] for p in document["passages"]])
            for document in documents or DOCUMENTS]


@pytest.fixture
def store():
    if not os.environ.get(TEST_DB.password_env):
        pytest.skip("CLM_TEST_POSTGRES_PASSWORD is not set")
    store = documents_pg.connect_postgres(TEST_DB, os.environ[TEST_DB.password_env])
    store.replace("cases", DOCUMENTS, vectors(), built())
    yield store
    store.replace("cases", [], [], built())
    store.connection.close()


def test_ollama_embedder_posts_the_texts():
    seen = []

    def handle(request):
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    embedder = OllamaEmbedder("http://ollama", http=httpx.Client(transport=httpx.MockTransport(handle)))
    assert embedder.embed("toy-embed", ["one", "two"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert seen == [("/api/embed", {"model": "toy-embed", "input": ["one", "two"]})]


def test_an_embedder_failure_is_one_line():
    embedder = OllamaEmbedder("http://ollama", http=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(404, json={"error": "model not found"}))))
    with pytest.raises(StepError, match="^embedding with toy-embed at http://ollama: "):
        embedder.embed("toy-embed", ["one"])


def test_the_embedder_keeps_one_client_with_a_timeout_for_searches(monkeypatch):
    """A search embeds one query: a stalled Ollama should not hold it for the ten minutes a batch load may take."""
    made, real = [], httpx.Client

    def client(timeout):
        made.append(timeout)
        return real(timeout=timeout, transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})))
    monkeypatch.setattr(documents.httpx, "Client", client)
    embedder = OllamaEmbedder("http://ollama")
    assert not made, "an environment whose search never runs makes no client"
    embedder.embed("toy-embed", ["one"])
    embedder.embed("toy-embed", ["two"])
    assert made == [60]
    OllamaEmbedder("http://ollama", timeout=600).embed("toy-embed", ["three"])
    assert made == [60, 600]


def test_an_unreachable_database_is_one_line():
    with pytest.raises(ScenarioError, match="^documents: cannot connect to 127.0.0.1:1/nowhere: ") as error:
        documents_pg.connect_postgres(NEVER, "x")
    assert "\n" not in str(error.value)


def test_no_vector_extension_is_one_line(monkeypatch):
    class Connection:
        """A connection whose database has no vector extension."""
        def execute(self, sql, *args):
            raise psycopg.errors.UndefinedFile('extension "vector" is not available\nDETAIL: more')

        def close(self):
            pass
    monkeypatch.setattr(psycopg, "connect", lambda **options: Connection())
    with pytest.raises(ScenarioError, match=r"^documents: 127.0.0.1:1/nowhere has no vector extension "
                                            r"\(brew install pgvector\): extension \"vector\" is not available$"):
        documents_pg.connect_postgres(NEVER, "x")


class FakeConnection:
    """A connection, and its own cursor, that fails a statement containing `fail[0]` with `fail[1]`."""
    def __init__(self, fail=None, rows=()):
        self.fail, self.rows, self.ran = fail, list(rows), []
        self.closed = self.broken = False

    def execute(self, sql, params=None):
        self.ran.append(sql)
        if self.fail and self.fail[0] in sql:
            self.broken = isinstance(self.fail[1], psycopg.OperationalError)
            raise self.fail[1]
        return self

    def executemany(self, sql, rows):
        self.execute(sql)

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True

    def transaction(self):
        return contextlib.nullcontext()

    def cursor(self):
        return contextlib.nullcontext(self)


@pytest.fixture
def opens(monkeypatch):
    """Connections handed out in turn by psycopg.connect, with the vector type registered on none of them."""
    waiting = []
    monkeypatch.setattr(psycopg, "connect", lambda **options: waiting.pop(0))
    monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda connection: None)
    return waiting


def test_tables_that_cannot_be_made_are_one_line(opens):
    """A role without the right to create tables fails at start-up in one line, not a traceback."""
    connection = FakeConnection(fail=("CREATE TABLE", psycopg.errors.InsufficientPrivilege(
        "permission denied for schema public\nDETAIL: more")))
    opens.append(connection)
    with pytest.raises(ScenarioError, match=r"^documents: 127.0.0.1:1/nowhere: cannot prepare the tables: "
                                            r"permission denied for schema public$"):
        documents_pg.connect_postgres(NEVER, "x")
    assert connection.closed


def test_a_failed_load_is_one_line():
    """PostgreSQL refuses the rows only once every passage has been embedded, so say why in one line."""
    store = documents_pg.PostgresStore(FakeConnection(fail=("INSERT INTO passages", psycopg.DataError(
        "PostgreSQL text fields cannot contain NUL (0x00) bytes"))), "127.0.0.1:1/nowhere")
    with pytest.raises(StepError, match=r"^documents at 127.0.0.1:1/nowhere: PostgreSQL text fields cannot"):
        store.replace("cases", DOCUMENTS, vectors(), built())


def test_a_lost_connection_is_opened_again(opens):
    """PostgreSQL restarted, or the laptop slept: the next statement reconnects once rather than failing forever."""
    lost, fresh = FakeConnection(fail=("SELECT id", psycopg.OperationalError("the connection is lost"))), \
        FakeConnection(rows=[("case-a",)])
    opens += [lost, fresh]
    store = documents_pg.connect_postgres(NEVER, "x")
    assert store.ids("cases", 3) == ["case-a"] and not opens
    assert store.ids("cases", 3) == ["case-a"] and store.connection is fresh


def test_a_replaced_connection_is_closed(opens):
    lost, fresh = FakeConnection(fail=("SELECT id", psycopg.OperationalError("the connection is lost"))), \
        FakeConnection(rows=[("case-a",)])
    opens += [lost, fresh]
    store = documents_pg.connect_postgres(NEVER, "x")
    assert store.ids("cases", 3) == ["case-a"] and lost.closed


def test_a_database_still_down_is_tried_once_per_read(opens, monkeypatch):
    """While PostgreSQL is down each read waits for one connect timeout, not two, before its one line."""
    tries = []

    def connect(**options):
        if opens:
            return opens.pop(0)
        tries.append(options)
        raise psycopg.OperationalError("connection refused")
    monkeypatch.setattr(psycopg, "connect", connect)
    opens.append(FakeConnection(fail=("SELECT id", psycopg.OperationalError("the connection is lost"))))
    store = documents_pg.connect_postgres(NEVER, "x")
    with pytest.raises(StepError, match=r"^documents at 127.0.0.1:1/nowhere: connection refused$"):
        store.ids("cases", 3)
    assert len(tries) == 1
    with pytest.raises(StepError, match=r"connection refused$"):
        store.ids("cases", 3)
    assert len(tries) == 2


def test_a_new_connection_that_cannot_take_vectors_is_closed(opens, monkeypatch):
    lost, fresh = FakeConnection(fail=("SELECT id", psycopg.OperationalError("the connection is lost"))), \
        FakeConnection()
    opens += [lost, fresh]

    def register_vector(connection):
        if connection is fresh:
            raise psycopg.ProgrammingError("vector type not found in the database")
    monkeypatch.setattr(pgvector.psycopg, "register_vector", register_vector)
    store = documents_pg.connect_postgres(NEVER, "x")
    with pytest.raises(StepError, match=r"vector type not found in the database$"):
        store.ids("cases", 3)
    assert fresh.closed


def test_a_failing_statement_is_not_run_again(opens):
    connection = FakeConnection(fail=("SELECT id", psycopg.errors.UndefinedTable('relation "documents" does not '
                                                                                 'exist')))
    opens += [connection, FakeConnection()]
    store = documents_pg.connect_postgres(NEVER, "x")
    with pytest.raises(StepError, match=r'^documents at 127.0.0.1:1/nowhere: relation "documents" does not exist$'):
        store.ids("cases", 3)
    assert store.connection is connection and connection.ran.count(connection.ran[-1]) == 1


def test_no_driver_names_the_documents_extra(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "psycopg", None)
    with pytest.raises(ScenarioError, match=r"install the documents extra"):
        documents_pg.connect_postgres(NEVER, "x")


@pytest.mark.postgres
def test_replace_then_search_finds_by_meaning_and_by_words(store):
    found = store.search("cases", FakeEmbedder().embed("toy-embed", ["reform covenant territory"])[0],
                         "reform covenant territory", {}, 3)
    assert found[0]["document"] == "case-a" and found[0]["passage"] == "2"
    assert set(found[0]) == {"document", "citation", "title", "metadata", "passage", "text", "score"}
    assert found[0]["citation"] == "100 S.W.3d 1" and found[0]["metadata"] == {"state": "TX", "year": 2019}
    # Words alone: the vector points nowhere near the statute, its wording does
    words = store.search("cases", FakeEmbedder().embed("toy-embed", ["zzz"])[0], "geographical", {}, 5)
    assert words[0]["document"] == "tx-15.50"


@pytest.mark.postgres
def test_filters_narrow_the_search(store):
    vector = FakeEmbedder().embed("toy-embed", ["restriction"])[0]
    assert {row["document"] for row in store.search("cases", vector, "restriction", {"state": "NY"}, 10)} == {"ny-1"}
    in_range = store.search("cases", vector, "restriction", {"year": {"from": 2020, "to": 2025}}, 10)
    assert {row["document"] for row in in_range} == {"case-b"}
    listed = store.search("cases", vector, "restriction", {"state": ["NY", "CA"]}, 10)
    assert {row["document"] for row in listed} == {"ny-1"}


@pytest.mark.postgres
def test_cited_by_walks_two_steps(store):
    assert store.cited_by("cases", "tx-15.50", 1) == [
        {"document": "case-a", "citation": "100 S.W.3d 1", "depth": 1, "passage": "2"}]
    assert [(row["document"], row["depth"]) for row in store.cited_by("cases", "tx-15.50", 2)] == [
        ("case-a", 1), ("case-b", 2)]
    assert [row["document"] for row in store.cites("cases", "case-b", 2)] == ["case-a", "tx-15.50"]
    # A citing passage belongs to the citing document, which a cites row does not name
    assert {row["passage"] for row in store.cites("cases", "case-b", 2)} == {None}


@pytest.mark.postgres
def test_document_gives_its_passages(store):
    assert store.document("cases", "case-a", "2")["passages"] == [
        {"label": "2", "text": "We reform the covenant to the territory he served."}]
    assert len(store.document("cases", "case-a", None)["passages"]) == 2
    assert store.document("cases", "nothing", None) == {}
    assert store.exists("cases", "tx-15.50") and not store.exists("cases", "nothing")


@pytest.mark.postgres
def test_ids_are_in_order_up_to_the_limit(store):
    assert store.ids("cases", 3) == ["case-a", "case-b", "ny-1"]
    assert store.ids("nothing", 3) == []


@pytest.mark.postgres
def test_replace_removes_what_was_there(store):
    assert store.built("cases")["documents"] == 4
    store.replace("cases", DOCUMENTS[:1], vectors(DOCUMENTS[:1]), built())
    assert store.built("cases") | {} == {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "toy-embed",
                                         "dimensions": 26, "documents": 1, "passages": 1}
    assert not store.exists("cases", "case-a")
    assert store.built("nothing") is None


@pytest.mark.postgres
def test_a_walk_of_a_dense_graph_visits_each_document_once(store):
    """Every one of 25 documents cites every other: walking paths, not documents, would give millions of rows."""
    dense = [{"id": f"d{n:02}", "citation": f"{n} S.W.3d 1", "title": f"D{n}", "metadata": {},
              "passages": [{"label": "1", "text": "We follow the others."}],
              "cites": [{"id": f"d{m:02}", "passage": "1"} for m in range(25) if m != n]} for n in range(25)]
    store.replace("cases", dense, vectors(dense), built())
    store.connection.execute("SET statement_timeout = '5s'")
    rows = store.cited_by("cases", "d00", 5)
    assert len(rows) == 24 and {row["depth"] for row in rows} == {1}
