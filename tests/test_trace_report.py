"""Tests for trace_report.py: the time window, the time parser, and the callers table and error count."""
# Test names say what they check.
# pylint: disable=missing-function-docstring
from datetime import datetime, timezone

import pytest

from clm_decision import trace_report


def record(started, caller=None, error=None, **extra):
    found = {"tool": "decide", "started": started, "counts": {}, "tools": [], "ms": 10.0,
             "outcome": {"error": error} if error else {"ok": True}, **extra}
    if caller is not None:
        found["caller"] = caller
    return found


def test_window_keeps_calls_started_inside():
    records = [record(t) for t in (90.0, 100.0, 150.0, 200.0, 210.0)]
    kept = trace_report.window(records, 100.0, 200.0)
    assert [r["started"] for r in kept] == [100.0, 150.0, 200.0]
    assert len(trace_report.window(records, None, None)) == 5
    assert [r["started"] for r in trace_report.window(records, 200.0, None)] == [200.0, 210.0]


def test_window_drops_synthetic():
    records = [record(1.0), record(2.0, synthetic=True)]
    assert [r["started"] for r in trace_report.window(records, None, None)] == [1.0]


def test_parse_time():
    assert trace_report.parse_time("15m", 10000.0) == 10000.0 - 900
    assert trace_report.parse_time("2h", 10000.0) == 10000.0 - 7200
    assert trace_report.parse_time("30s", 10000.0) == 10000.0 - 30
    assert trace_report.parse_time("2026-11-06T12:00:00+00:00", 0.0) == datetime(
        2026, 11, 6, 12, tzinfo=timezone.utc).timestamp()


def test_parse_time_refuses_junk():
    with pytest.raises(ValueError):
        trace_report.parse_time("yesterday", 0.0)


def test_report_has_callers_and_errors():
    records = [record(1.0, {"client_id": "desk"}), record(2.0, {"client_id": "desk"}, error="ClmError"),
               record(3.0, {"key_id": "ab12cd34"}), record(4.0)]
    text = trace_report.report(trace_report.summarise(records), records)
    assert "desk" in text and "ab12cd34" in text and "unknown" in text
    desk = next(line for line in text.splitlines() if line.strip().startswith("desk"))
    assert desk.split()[1:] == ["2", "1"]
    assert "errors: 1" in text


def test_main_says_when_the_window_has_no_calls(tmp_path, capsys):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"tool": "decide", "started": 1.0, "counts": {}, "tools": [], "ms": 1.0, '
                    '"outcome": {"ok": true}}\n')
    assert trace_report.main([str(path), "--since", "5m"]) == 0
    assert "no calls between" in capsys.readouterr().out
    assert trace_report.main([str(path), "--since", "junk"]) == 2
