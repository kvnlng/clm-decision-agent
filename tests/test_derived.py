"""Tests for the derived layer's file: reading it, writing it, and changing an entry's status."""
# Test names say what they check.
# pylint: disable=missing-function-docstring
import json

import pytest

from clm_decision import derived


def _layer() -> dict:
    layer = derived.empty("handbook")
    layer["decisions"]["carryover_limit"] = {"status": "checked", "check": {"passed": True, "failures": []}}
    layer["facts"]["k401_2026.match_cap"] = {"status": "proposed", "check": {"passed": False,
                                                                             "failures": ["6300 not in § 4(2)"]}}
    layer["conflicts"]["k401_2026.match_cap"] = {"status": "open"}
    return layer


def test_a_folder_with_no_layer_reads_as_empty(tmp_path):
    assert derived.read(tmp_path) == derived.empty(None)


def test_written_layer_reads_back(tmp_path):
    derived.write(tmp_path, _layer())
    assert derived.read(tmp_path) == _layer()
    assert json.loads((tmp_path / "derived" / "derived.json").read_text(encoding="utf-8"))["version"] == 1


def test_approving_a_checked_entry_records_who_and_when(tmp_path):
    derived.write(tmp_path, _layer())
    entry = derived.set_status(tmp_path, "decisions", "carryover_limit", "approved", by="kevin", note="matches § 3")
    assert entry["status"] == "approved"
    assert entry["decided_by"] == "kevin" and entry["note"] == "matches § 3" and entry["decided_at"]
    assert derived.read(tmp_path)["decisions"]["carryover_limit"]["status"] == "approved"


def test_an_entry_that_failed_its_check_cannot_be_approved(tmp_path):
    derived.write(tmp_path, _layer())
    with pytest.raises(derived.DerivedError, match="failed its check"):
        derived.set_status(tmp_path, "facts", "k401_2026.match_cap", "approved", by="kevin")


def test_any_entry_can_be_rejected(tmp_path):
    derived.write(tmp_path, _layer())
    assert derived.set_status(tmp_path, "facts", "k401_2026.match_cap", "rejected", by="kevin")["status"] == "rejected"


def test_a_conflict_is_resolved_not_approved(tmp_path):
    derived.write(tmp_path, _layer())
    entry = derived.set_status(tmp_path, "conflicts", "k401_2026.match_cap", "resolved", by="kevin",
                               note="the supplied cap is current")
    assert entry["status"] == "resolved"
    with pytest.raises(derived.DerivedError, match="not a status"):
        derived.set_status(tmp_path, "conflicts", "k401_2026.match_cap", "approved", by="kevin")


def test_unknown_entries_and_statuses_are_refused(tmp_path):
    derived.write(tmp_path, _layer())
    with pytest.raises(derived.DerivedError, match="no decisions entry"):
        derived.set_status(tmp_path, "decisions", "nope", "approved", by="kevin")
    with pytest.raises(derived.DerivedError, match="not a status"):
        derived.set_status(tmp_path, "decisions", "carryover_limit", "fine", by="kevin")
    with pytest.raises(derived.DerivedError, match="by"):
        derived.set_status(tmp_path, "decisions", "carryover_limit", "approved", by="")
