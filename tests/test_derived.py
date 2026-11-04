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


def test_a_proposed_entry_cannot_be_approved_even_with_an_old_passing_check(tmp_path):
    # A correction sets the entry back to proposed; until clm-derive check runs, its old check proves nothing
    layer = _layer()
    layer["decisions"]["carryover_limit"]["status"] = "proposed"
    derived.write(tmp_path, layer)
    with pytest.raises(derived.DerivedError, match="checked"):
        derived.set_status(tmp_path, "decisions", "carryover_limit", "approved", by="kevin")


def test_a_rejected_entry_that_passed_its_check_can_still_be_approved(tmp_path):
    derived.write(tmp_path, _layer())
    derived.set_status(tmp_path, "decisions", "carryover_limit", "rejected", by="kevin")
    assert derived.set_status(tmp_path, "decisions", "carryover_limit", "approved", by="kevin")["status"] == "approved"


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


def test_set_check_moves_a_passing_proposed_entry_to_checked_and_records_who(tmp_path):
    layer = _layer()
    layer["decisions"]["carryover_limit"].update(status="proposed", check=None)
    derived.write(tmp_path, layer)
    entry = derived.set_check(tmp_path, "decisions", "carryover_limit", {"passed": True, "failures": []})
    assert entry["status"] == "checked"
    assert entry["check"]["by"] == "clm-derive check" and entry["check"]["at"] and entry["check"]["passed"]
    assert derived.read(tmp_path)["decisions"]["carryover_limit"]["status"] == "checked"


def test_set_check_moves_a_failing_checked_entry_back_to_proposed(tmp_path):
    derived.write(tmp_path, _layer())
    entry = derived.set_check(tmp_path, "decisions", "carryover_limit", {"passed": False, "failures": ["no 7"]})
    assert entry["status"] == "proposed" and entry["check"]["failures"] == ["no 7"]


def test_set_check_never_touches_approved_or_rejected_entries(tmp_path):
    derived.write(tmp_path, _layer())
    derived.set_status(tmp_path, "decisions", "carryover_limit", "approved", by="kevin", note="ok")
    with pytest.raises(derived.DerivedError, match="approved"):
        derived.set_check(tmp_path, "decisions", "carryover_limit", {"passed": False, "failures": ["x"]})
    derived.set_status(tmp_path, "facts", "k401_2026.match_cap", "rejected", by="kevin")
    with pytest.raises(derived.DerivedError, match="rejected"):
        derived.set_check(tmp_path, "facts", "k401_2026.match_cap", {"passed": True, "failures": []})
    layer = derived.read(tmp_path)
    assert layer["decisions"]["carryover_limit"]["status"] == "approved"
    assert layer["decisions"]["carryover_limit"]["decided_by"] == "kevin"


def test_set_check_keeps_the_decision_fields(tmp_path):
    layer = _layer()
    layer["decisions"]["carryover_limit"].update(decided_by="kevin", decided_at="2026-11-04", note="corrected")
    derived.write(tmp_path, layer)
    entry = derived.set_check(tmp_path, "decisions", "carryover_limit", {"passed": True, "failures": []})
    assert (entry["decided_by"], entry["decided_at"], entry["note"]) == ("kevin", "2026-11-04", "corrected")


def test_set_stale_sends_an_approved_entry_back_to_proposed_keeping_who_decided(tmp_path):
    derived.write(tmp_path, _layer())
    derived.set_status(tmp_path, "decisions", "carryover_limit", "approved", by="kevin", note="matches § 5")
    decided_at = derived.read(tmp_path)["decisions"]["carryover_limit"]["decided_at"]
    entry = derived.set_stale(tmp_path, "decisions", "carryover_limit", "the cited document has a newer edition")
    assert entry["status"] == "proposed" and entry["check"]["passed"] is False
    assert entry["check"]["failures"] == ["the cited document has a newer edition"]
    assert entry["check"]["by"] == "clm-derive check"
    assert (entry["decided_by"], entry["decided_at"]) == ("kevin", decided_at)
    assert entry["note"].startswith("matches § 5") and "newer edition" in entry["note"]
    assert derived.read(tmp_path)["decisions"]["carryover_limit"]["status"] == "proposed"


def test_set_stale_never_touches_a_rejected_entry(tmp_path):
    derived.write(tmp_path, _layer())
    derived.set_status(tmp_path, "facts", "k401_2026.match_cap", "rejected", by="kevin")
    with pytest.raises(derived.DerivedError, match="rejected"):
        derived.set_stale(tmp_path, "facts", "k401_2026.match_cap", "newer edition")


def test_set_check_refuses_conflicts(tmp_path):
    derived.write(tmp_path, _layer())
    with pytest.raises(derived.DerivedError, match="decisions or facts"):
        derived.set_check(tmp_path, "conflicts", "k401_2026.match_cap", {"passed": True, "failures": []})
