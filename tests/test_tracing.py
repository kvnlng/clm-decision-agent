"""Tests for tracing.py, trace_report.py and the tracing they switch on in the server and the engines."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
import json
import threading

import numpy as np
import pytest
from conftest import BASE_TOML, FakeRouter, decision
from mcp import Client
from test_clm_torch import CONFIG, FakeEncoder, StubTokenizer, tiny_qwen, tiny_weights

import clm_router
import server
import trace_report
import tracing
from clm_common import ClmError
from clm_router import LocalContrastiveRouter, MlxEngine
from clm_torch import TorchEncoder, TorchEngine, TorchHeads
from scenario import load_scenario
from tools import load_tools

TEXT = "charged twice"
URGENT = {"q": {"type": "noul", "instructions": "Urgent?"}}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def lines(path, writer=None):
    """The trace's records; `writer` is closed first, so its background thread has written everything."""
    if writer is not None:
        writer.close()
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_traced_call_writes_one_line_with_its_counts(tmp_path):
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    with writer.call("decide", TEXT):
        tracing.add("cache_hits", 2)
        tracing.add("cache_hits")
        with tracing.timed("encode_ms"):
            pass
    [record] = lines(tmp_path / "trace.jsonl", writer)
    assert record["tool"] == "decide"
    assert record["counts"]["cache_hits"] == 3
    assert record["counts"]["encode_ms"] >= 0
    assert record["ms"] >= record["counts"]["encode_ms"]
    assert isinstance(record["started"], float)


def test_a_trace_leaves_out_the_text_unless_asked(tmp_path):
    plain, with_text = tracing.TraceWriter(tmp_path / "plain.jsonl"), tracing.TraceWriter(tmp_path / "text.jsonl",
                                                                                          text=True)
    for writer in (plain, with_text):
        with writer.call("decide", TEXT):
            pass
    assert "text" not in lines(tmp_path / "plain.jsonl", plain)[0]
    assert lines(tmp_path / "text.jsonl", with_text)[0]["text"] == TEXT


def test_counting_outside_a_traced_call_does_nothing(tmp_path):
    assert not tracing.active()
    tracing.add("cache_hits")
    with tracing.timed("encode_ms"):
        pass
    tracing.note_tool("lookup", "inform", 1.0, "done")
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    with writer.call("decide"):
        assert tracing.active()
    assert lines(tmp_path / "trace.jsonl", writer)[0]["counts"] == {}


def test_a_trace_that_cannot_be_written_leaves_calls_working(tmp_path, capsys):
    (tmp_path / "taken").write_text("a file where the folder should be")
    writer = tracing.TraceWriter(tmp_path / "taken" / "trace.jsonl")
    for _ in range(3):
        with writer.call("decide"):
            pass
    writer.close()
    assert capsys.readouterr().err.count("tracing stopped") == 1


def test_a_count_json_cannot_hold_stops_tracing_not_the_call(tmp_path, capsys):
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    with writer.call("decide"):
        tracing.add("odd", np.float16(1))
    with writer.call("decide"):
        pass
    writer.close()
    assert "tracing stopped" in capsys.readouterr().err


def test_a_failed_call_is_still_written(tmp_path):
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    with pytest.raises(ValueError), writer.call("decide"):
        raise ValueError("no")
    assert lines(tmp_path / "trace.jsonl", writer)[0]["outcome"] == {"error": "ValueError"}


def test_calls_on_threads_are_traced_separately(tmp_path):
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    barrier = threading.Barrier(2)

    def work(n):
        with writer.call(f"call{n}"):
            barrier.wait()
            tracing.add("cache_hits", n)
    threads = [threading.Thread(target=work, args=(n,)) for n in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert {r["tool"]: r["counts"]["cache_hits"] for r in lines(tmp_path / "trace.jsonl", writer)} == \
        {"call1": 1, "call2": 2}


@pytest.fixture
def traced_server(write_scenario, tmp_path):
    """-> (server for BASE_TOML with tracing on, the trace path). CLM answers Billing, urgent, so all tools run."""
    scenario = load_scenario(write_scenario(BASE_TOML))
    urgent = {"urgent": {"type": "noul", "noul": 0.9}}
    router = FakeRouter(decision("Billing", answers=urgent), asks=[urgent])
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    return server.build_server(scenario, router, load_tools(scenario), review=False, trace=writer), writer


@pytest.mark.anyio
async def test_decide_is_traced_with_its_tools_and_outcome(traced_server):
    mcp_server, writer = traced_server
    async with Client(mcp_server) as client:
        await client.call_tool("decide", {"text": TEXT})
    [record] = lines(writer.path, writer)
    assert record["tool"] == "decide"
    assert [(t["tool"], t["kind"], t["status"]) for t in record["tools"]] == \
        [("lookup", "inform", "done"), ("act", "act", "done")]
    assert all(t["ms"] >= 0 for t in record["tools"])
    assert record["outcome"] == {"route": "Billing", "confidence": 0.9, "uncertain": False, "source": "clm"}
    assert record["counts"]["lock_wait_ms"] >= 0
    assert "text" not in record


@pytest.mark.anyio
async def test_check_call_and_describe_are_traced(traced_server):
    mcp_server, writer = traced_server
    async with Client(mcp_server) as client:
        await client.call_tool("check_call", {"text": TEXT, "call": {"tool": "refund"}})
        await client.call_tool("describe", {})
    check, describe = lines(writer.path, writer)
    assert (check["tool"], check["outcome"]) == ("check_call", {"p": 0.9, "approved": True})
    assert describe["tool"] == "describe"


def test_the_torch_engine_counts_cache_texts_tokens_and_padding(tmp_path):
    engine = TorchEngine(FakeEncoder(), TorchHeads(tiny_weights(), CONFIG, "cpu"))
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    with writer.call("first"):
        engine.answer("a", URGENT)
    with writer.call("second"):
        engine.answer("a", URGENT)
    with writer.call("duplicates"):
        engine._vectors(["new", "new"], "state")  # pylint: disable=protected-access
    first, second, duplicates = (r["counts"] for r in lines(tmp_path / "trace.jsonl", writer))
    # One state text and the noul question's two option texts
    assert (first["clm_calls"], first["cache_hits"], first["cache_misses"]) == (1, 0, 3)
    assert (second["cache_hits"], second["cache_misses"]) == (3, 0)
    assert first["head_ms"] >= 0 and "head_ms" not in second
    # The second "new" is served by the first one's encoding
    assert (duplicates["cache_hits"], duplicates["cache_misses"]) == (1, 1)

    encoder = TorchEncoder("test/encoder", None, "cpu", loaded=(tiny_qwen(), StubTokenizer()))
    writer = tracing.TraceWriter(tmp_path / "encode.jsonl")
    with writer.call("encode"):
        encoder.embed(["ab", "abcd"])
    counts = lines(tmp_path / "encode.jsonl", writer)[0]["counts"]
    # Both texts packed in one row, with no padding
    assert (counts["encoder_batches"], counts["texts_encoded"], counts["tokens"], counts["padded_tokens"]) == \
        (1, 2, 6, 6)
    assert counts["encode_ms"] >= 0


class FakeMlxEncoder:
    def embed(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32), 5 * len(texts)


class FakeMlxEngine:
    """The MLX build's Engine as far as MlxEngine uses it: a cache keyed by (which, text) and a token count."""

    def __init__(self):
        self.cache, self.tokens_spent, self.encoder = {}, 0, FakeMlxEncoder()

    def answer(self, state, questions):
        self.tokens_spent += 7
        self.cache[("state", f"{state}\n\n{questions['q']['instructions']}")] = 1
        return {"answers": {}}


class FakeMlxBuild(FakeMlxEngine):
    """Also encodes in batches, at four tokens a text, as prefetch uses the build."""

    def _vectors(self, texts, which):
        self.tokens_spent += 4 * len({text for text in texts if (which, text) not in self.cache})
        self.cache.update({(which, text): 1 for text in texts})


def test_the_mlx_engine_counts_cache_texts_and_tokens(tmp_path):
    engine = MlxEngine.__new__(MlxEngine)
    engine.engine = FakeMlxEngine()
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    with writer.call("first"):
        engine.answer("a", URGENT)
    with writer.call("second"):
        engine.answer("a", URGENT)
    with writer.call("embed"):
        engine.embed(["a", "b"])
    first, second, embed = (r["counts"] for r in lines(tmp_path / "trace.jsonl", writer))
    assert (first["clm_calls"], first["cache_misses"], first["tokens"]) == (1, 3, 7)
    assert (second["cache_hits"], second["cache_misses"]) == (1, 2)
    assert (embed["texts_encoded"], embed["tokens"]) == (2, 10)


def test_the_mlx_engine_skips_the_cache_check_when_not_tracing(monkeypatch):
    engine = MlxEngine.__new__(MlxEngine)
    engine.engine = FakeMlxEngine()
    monkeypatch.setattr(clm_router.clm_schema, "build_pairs", None)
    engine.answer("a", URGENT)


def test_matching_corrections_is_timed(tmp_path):
    router = LocalContrastiveRouter(engine=None)
    router.verdicts = [([1.0, 0.0], "Billing")]
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    with writer.call("decide"):
        assert router.match_verdict([1.0, 0.0], 0.9) == ("Billing", 1.0)
    assert lines(tmp_path / "trace.jsonl", writer)[0]["counts"]["match_ms"] >= 0


def record(started, ms, tool="decide", tools=(), **counts):
    return {"tool": tool, "started": started, "ms": ms, "counts": counts,
            "tools": [{"tool": name, "kind": kind, "ms": t, "status": "done"} for name, kind, t in tools],
            "outcome": {}}


def test_the_report_finds_where_the_time_went():
    records = [record(100.0 + i, 10.0, tools=[("lookup", "inform", 2.0), ("act", "act", 1.0)],
                      lock_wait_ms=1.0, encode_ms=4.0, head_ms=1.0, cache_hits=3, cache_misses=1,
                      texts_encoded=1, tokens=20, padded_tokens=25)
               for i in range(10)]
    records.append(record(105.0, 2.0, tool="describe"))
    summary = trace_report.summarise(records)
    decide = summary["decide"]
    assert decide["calls"] == 10
    # Ten calls, the first starting at 100 s and the last ending at 109.01 s
    assert decide["per_second"] == pytest.approx(10 / 9.01)
    assert decide["stages"]["encode"]["p50"] == 4.0
    assert decide["stages"]["inform tools"]["share"] == pytest.approx(0.2)
    assert decide["stages"]["other"]["p50"] == pytest.approx(1.0)
    assert decide["cache_hit_rate"] == pytest.approx(0.75)
    assert decide["texts_per_call"] == 1.0
    assert decide["padding"] == pytest.approx(0.2)
    assert summary["describe"]["calls"] == 1


def test_the_report_prints_a_table(tmp_path, capsys):
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(record(100.0 + i, 10.0, encode_ms=4.0)) for i in range(3)) + "\n")
    assert trace_report.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "decide: 3 calls" in out and "encode" in out


def test_the_report_skips_a_last_line_cut_off_mid_write(tmp_path, capsys):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(record(100.0, 10.0)) + "\n" + '{"tool": "dec')
    assert trace_report.main([str(path)]) == 0
    assert "decide: 1 calls" in capsys.readouterr().out


def test_the_report_explains_a_missing_file(tmp_path, capsys):
    assert trace_report.main([str(tmp_path / "missing.jsonl")]) == 2
    assert "error:" in capsys.readouterr().err


def test_the_server_takes_trace_options(write_scenario, monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(server, "load", lambda path: ("scenario", "router", {}))
    monkeypatch.setattr(server, "build_server", lambda *args, **kwargs: seen.update(kwargs) or FakeServer())
    server.main([str(write_scenario()), "--trace", str(tmp_path / "t.jsonl"), "--trace-text"])
    assert seen["trace"].path == tmp_path / "t.jsonl" and seen["trace"].text is True


class FakeServer:
    def run(self, transport):
        pass


@pytest.mark.anyio
async def test_a_failing_clm_or_tool_is_traced_and_reported(write_scenario, tmp_path):
    scenario = load_scenario(write_scenario(BASE_TOML.replace('tool = "act"', 'tool = "broken"')))
    urgent = {"urgent": {"type": "noul", "noul": 0.9}}
    router = FakeRouter(decision("Billing", answers=urgent), asks=[urgent])
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    mcp_server = server.build_server(scenario, router, load_tools(scenario), review=False, trace=writer)

    def no_clm(*args):
        raise ClmError("CLM is not loaded")
    async with Client(mcp_server) as client:
        await client.call_tool("decide", {"text": TEXT})
        router.check_tool = no_clm
        failed = await client.call_tool("check_call", {"text": TEXT, "call": {}})
    assert failed.is_error
    decided, checked = lines(writer.path, writer)
    assert decided["tools"][-1]["status"] == "failed"
    assert checked["outcome"] == {"error": "ToolError"}


@pytest.mark.anyio
async def test_without_tracing_decide_reads_only_what_it_returns(write_scenario):
    scenario = load_scenario(write_scenario())
    found = decision("Tech")
    del found["source"]
    async with Client(server.build_server(scenario, FakeRouter(found), {}, review=False)) as client:
        assert not (await client.call_tool("decide", {"text": TEXT})).is_error


def test_the_mlx_prefetch_is_counted(tmp_path):
    engine = MlxEngine.__new__(MlxEngine)
    engine.engine = FakeMlxBuild()
    writer = tracing.TraceWriter(tmp_path / "trace.jsonl")
    with writer.call("decide_many"):
        engine.prefetch(["a", "b"], URGENT)
    counts = lines(tmp_path / "trace.jsonl", writer)[0]["counts"]
    # Two state texts and the question's two option texts, at four tokens each
    assert (counts["texts_encoded"], counts["tokens"]) == (4, 16)


def test_the_report_counts_decisions_for_decide_many():
    records = [record(100.0, 1000.0, tool="decide_many", decisions=50), record(101.0, 1000.0, tool="decide_many",
                                                                               decisions=50)]
    summary = trace_report.summarise(records)["decide_many"]
    assert summary["decisions"] == 100 and summary["per_second"] == pytest.approx(100 / 2.0)
    assert "100 decisions" in trace_report.report({"decide_many": summary})


def test_the_trace_file_s_folder_is_made_if_missing(tmp_path):
    writer = tracing.TraceWriter(tmp_path / "runs" / "trace.jsonl")
    with writer.call("decide", TEXT):
        pass
    assert len(lines(tmp_path / "runs" / "trace.jsonl", writer)) == 1
