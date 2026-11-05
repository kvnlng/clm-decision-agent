"""Tests for load_documents.py: JSON lines checked, split into passages, embedded and loaded into a collection."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json

import pytest
from fake_documents import FakeEmbedder, FakeStore

from clm_decision import load_documents
from clm_decision.documents import Collection

CASES = Collection("cases", "Opinions", {"state": "string", "year": "integer", "decided": "date"}, "toy-embed")
GOOD = {"id": "case-a", "citation": "100 S.W.3d 1", "title": "Case A",
        "metadata": {"state": "TX", "year": 2019, "decided": "2019-05-01"},
        "text": "The manager was barred.\n\nWe reform the covenant.", "cites": ["tx-15.50"]}


def write(tmp_path, *lines):
    path = tmp_path / "cases.jsonl"
    path.write_text("".join((line if isinstance(line, str) else json.dumps(line)) + "\n" for line in lines))
    return path


def test_a_good_line_is_read_and_its_text_split(tmp_path):
    documents = load_documents.read_jsonl(write(tmp_path, GOOD), CASES)
    assert len(documents) == 1
    document = documents[0]
    assert document["passages"] == [{"label": "1", "text": "The manager was barred."},
                                    {"label": "2", "text": "We reform the covenant."}]
    assert document["cites"] == [{"id": "tx-15.50", "passage": None}]


def test_a_bad_line_names_file_and_line(tmp_path):
    path = write(tmp_path, GOOD, "{not json")
    with pytest.raises(ValueError, match=r"^cases\.jsonl:2: not valid JSON"):
        load_documents.read_jsonl(path, CASES)
    with pytest.raises(ValueError, match=r"^cases\.jsonl:1: needs citation, as text"):
        load_documents.read_jsonl(write(tmp_path, {k: v for k, v in GOOD.items() if k != "citation"}), CASES)


def test_metadata_outside_the_declared_filters_is_refused(tmp_path):
    line = GOOD | {"metadata": GOOD["metadata"] | {"judge": "Smith"}}
    with pytest.raises(ValueError, match=r"^cases\.jsonl:1: metadata\.judge is not a filter of cases "
                                         r"\(decided, state, year\)"):
        load_documents.read_jsonl(write(tmp_path, line), CASES)


def test_metadata_of_the_wrong_type_is_refused(tmp_path):
    for field, value in (("year", "2019"), ("decided", "May 2019"), ("state", 3)):
        line = GOOD | {"metadata": GOOD["metadata"] | {field: value}}
        with pytest.raises(ValueError, match=rf"^cases\.jsonl:1: metadata\.{field} must be "):
            load_documents.read_jsonl(write(tmp_path, line), CASES)


def test_a_date_must_be_written_yyyy_mm_dd(tmp_path):
    """Search filters and in_force compare dates as YYYY-MM-DD text, so 20190501 or 2019-W18-3 would never match."""
    for value in ("20190501", "2019-W18-3", "2019-13-45"):
        line = GOOD | {"metadata": GOOD["metadata"] | {"decided": value}}
        with pytest.raises(ValueError, match=r"^cases\.jsonl:1: metadata\.decided must be a YYYY-MM-DD date"):
            load_documents.read_jsonl(write(tmp_path, line), CASES)


def test_sources_may_go_beside_the_filters(tmp_path):
    line = GOOD | {"metadata": GOOD["metadata"] | {"sources": {"year": "1"}}}
    assert load_documents.read_jsonl(write(tmp_path, line), CASES)[0]["metadata"]["sources"] == {"year": "1"}


def test_given_passages_are_kept_with_their_labels(tmp_path):
    line = {k: v for k, v in GOOD.items() if k != "text"} | {"passages": [{"label": "(a)", "text": "First."},
                                                                         {"label": "(c)", "text": "Third."}]}
    assert [p["label"] for p in load_documents.read_jsonl(write(tmp_path, line), CASES)[0]["passages"]] == [
        "(a)", "(c)"]


def test_a_repeated_id_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"^cases\.jsonl:2: id case-a is already on line 1"):
        load_documents.read_jsonl(write(tmp_path, GOOD, GOOD), CASES)


def test_a_long_paragraph_is_split_near_the_size():
    text = " ".join(["word"] * 1000)
    passages = load_documents.split(text)
    assert len(passages) > 1 and all(len(p["text"]) <= load_documents.PASSAGE_CHARS for p in passages)
    assert " ".join(p["text"] for p in passages) == text


def test_load_reports_built_at_and_counts(tmp_path):
    store, embedder = FakeStore(), FakeEmbedder()
    documents = load_documents.read_jsonl(write(tmp_path, GOOD), CASES)
    counts = load_documents.load(store, embedder, CASES, documents, source="test cases",
                                 now="2026-11-03T12:00:00Z")
    assert counts == {"documents": 1, "passages": 2, "citations": 1, "outside": 1,
                      "built_at": "2026-11-03T12:00:00Z"}
    assert store.built("cases") == {"built_at": "2026-11-03T12:00:00Z", "source": "test cases",
                                    "model": "toy-embed", "dimensions": 26}
    assert embedder.asked == [("toy-embed", ["The manager was barred.", "We reform the covenant."])]


def test_citations_outside_the_collection_are_counted(tmp_path):
    other = {"id": "tx-15.50", "citation": "Tex. Bus. & Com. Code § 15.50", "title": "Statute",
             "metadata": {"state": "TX"}, "text": "Reasonable limits."}
    documents = load_documents.read_jsonl(write(tmp_path, GOOD, other), CASES)
    counts = load_documents.load(FakeStore(), FakeEmbedder(), CASES, documents, source="x", now="t")
    assert counts["citations"] == 1 and counts["outside"] == 0


def test_main_names_an_unknown_collection(tmp_path, capsys):
    toml = tmp_path / "environment.toml"
    toml.write_text('[documents]\n[collections.cases]\ndescription = "x"\nembedding_model = "m"\n')
    assert load_documents.main([str(toml), "laws", str(write(tmp_path, GOOD))]) == 2
    assert capsys.readouterr().err == "error: no collection laws in environment.toml (cases)\n"


def test_a_nul_character_is_refused_before_embedding(tmp_path):
    """PostgreSQL text cannot hold NUL, so it would fail only after every passage had been embedded."""
    for line in (GOOD | {"text": "The manager\u0000 was barred."}, GOOD | {"title": "Case\u0000A"},
                 GOOD | {"metadata": GOOD["metadata"] | {"state": "T\u0000X"}},
                 {k: v for k, v in GOOD.items() if k != "text"} | {"passages": [{"label": "(a\u0000)", "text": "x"}]}):
        with pytest.raises(ValueError, match=r"^cases\.jsonl:1: .* contains a NUL character"):
            load_documents.read_jsonl(write(tmp_path, line), CASES)


def test_a_citing_passage_must_be_text(tmp_path):
    with pytest.raises(ValueError, match=r"^cases\.jsonl:1: each of cites is an id"):
        load_documents.read_jsonl(write(tmp_path, GOOD | {"cites": [{"id": "tx-15.50", "passage": 3}]}), CASES)


def test_a_file_saved_with_a_byte_order_mark_is_read(tmp_path):
    path = write(tmp_path, GOOD)
    path.write_text("\ufeff" + path.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_documents.read_jsonl(path, CASES)[0]["id"] == "case-a"
