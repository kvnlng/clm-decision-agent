"""Tests for the derived layer's file: reading it, writing it, and changing an entry's status."""
# Test names say what they check.
# pylint: disable=missing-function-docstring
import hashlib
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
    with pytest.raises(derived.DerivedError, match="no decisions entry named 'nope': it is not derived"):
        derived.set_status(tmp_path, "decisions", "nope", "approved", by="kevin")
    with pytest.raises(derived.DerivedError, match="no conflicts entry named 'nope'$"):
        derived.set_status(tmp_path, "conflicts", "nope", "resolved", by="kevin")
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


def test_set_check_records_a_second_reading_and_only_when_one_was_made(tmp_path):
    derived.write(tmp_path, _layer())
    reading = {"model": "m", "agreed": False, "differences": [{"first": 7, "second": 10}]}
    entry = derived.set_check(tmp_path, "decisions", "carryover_limit",
                              {"passed": False, "failures": ["x"], "second_reading": reading})
    assert entry["check"]["second_reading"] == reading
    entry = derived.set_check(tmp_path, "facts", "k401_2026.match_cap", {"passed": True, "failures": []})
    assert "second_reading" not in entry["check"]


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


def _with_table(folder) -> None:
    layer = _layer()
    layer["decisions"]["carryover_limit"] = {"dmn": "rules/carryover_limit.dmn", "status": "proposed", "check": None}
    derived.write(folder, layer)
    (folder / "derived" / "rules").mkdir()
    (folder / "derived" / "rules" / "carryover_limit.dmn").write_text("<table/>", encoding="utf-8")


def test_set_check_records_the_digest_of_the_table_it_checked(tmp_path):
    _with_table(tmp_path)
    entry = derived.set_check(tmp_path, "decisions", "carryover_limit", {"passed": True, "failures": []})
    assert entry["check"]["dmn_sha256"] == hashlib.sha256(b"<table/>").hexdigest()
    assert "dmn_sha256" not in derived.set_check(tmp_path, "facts", "k401_2026.match_cap", {"passed": True})["check"]
    assert not derived.changed_since_check(tmp_path, entry)


def test_a_table_changed_since_its_check_cannot_be_approved(tmp_path):
    _with_table(tmp_path)
    derived.set_check(tmp_path, "decisions", "carryover_limit", {"passed": True, "failures": []})
    (tmp_path / "derived" / "rules" / "carryover_limit.dmn").write_text("<edited/>", encoding="utf-8")
    with pytest.raises(derived.DerivedError, match="^carryover_limit's DMN has changed since its check: run "
                                                   "clm-derive check again$"):
        derived.set_status(tmp_path, "decisions", "carryover_limit", "approved", by="kevin")
    assert derived.changed_since_check(tmp_path, derived.read(tmp_path)["decisions"]["carryover_limit"])


def test_a_check_recorded_before_digests_were_kept_is_taken_as_it_is(tmp_path):
    _with_table(tmp_path)
    assert not derived.changed_since_check(tmp_path, {"dmn": "rules/carryover_limit.dmn",
                                                      "check": {"passed": True, "failures": []}})


def _read_again(folder, agreed=True, at="2026-11-04T10:00:00+00:00", **entry) -> None:
    layer = _layer()
    layer["decisions"]["carryover_limit"] = {
        "status": "checked", "decided_by": None, "decided_at": None, "note": None, **entry,
        "check": {"passed": agreed, "failures": [], "at": at, "by": "clm-derive check",
                  "second_reading": {"model": "second", "agreed": agreed, "differences": []}}}
    derived.write(folder, layer)


def approve(folder, reader="second"):
    return derived.set_status(folder, "decisions", "carryover_limit", "approved", by="kevin", second_reader=reader)


def test_with_a_second_reader_an_entry_is_approved_after_an_agreeing_reading(tmp_path):
    _read_again(tmp_path)
    assert approve(tmp_path)["status"] == "approved"


def test_with_a_second_reader_an_entry_never_read_again_is_not_approved(tmp_path):
    derived.write(tmp_path, _layer())
    with pytest.raises(derived.DerivedError, match="^carryover_limit has no second reading by second that agrees, "
                                                   "made since its last change: run clm-derive check"):
        approve(tmp_path)
    assert approve(tmp_path, reader=None)["status"] == "approved"


def test_with_a_second_reader_a_rejected_entry_with_an_old_reading_is_not_approved(tmp_path):
    _read_again(tmp_path, status="rejected", decided_by="kevin", decided_at="2026-11-04T11:00:00+00:00")
    with pytest.raises(derived.DerivedError, match="since its last change"):
        approve(tmp_path)
    # Set back to proposed by a person, then checked and read again: the reading is later than the change
    _read_again(tmp_path, at="2026-11-04T12:00:00+00:00", decided_by="kevin", decided_at="2026-11-04T11:00:00+00:00")
    assert approve(tmp_path)["status"] == "approved"
    # Set to checked by hand after an older reading: the reading is earlier than the change
    _read_again(tmp_path, at="2026-11-04T10:00:00+00:00", decided_by="kevin", decided_at="2026-11-04T11:00:00+00:00")
    with pytest.raises(derived.DerivedError, match="since its last change"):
        approve(tmp_path)


def test_set_check_refuses_conflicts(tmp_path):
    derived.write(tmp_path, _layer())
    with pytest.raises(derived.DerivedError, match="decisions or facts"):
        derived.set_check(tmp_path, "conflicts", "k401_2026.match_cap", {"passed": True, "failures": []})


@pytest.mark.parametrize("kind, name", [("decisions", "carryover_limit"), ("facts", "k401_2026.match_cap")])
def test_a_person_cannot_set_checked_only_clm_derive_check_does(tmp_path, kind, name):
    derived.write(tmp_path, _layer())
    derived.set_status(tmp_path, kind, name, "rejected", by="kevin")
    with pytest.raises(derived.DerivedError, match="checked is set only by clm-derive check"):
        derived.set_status(tmp_path, kind, name, "checked", by="kevin")
    assert derived.read(tmp_path)[kind][name]["status"] == "rejected"
    assert derived.set_status(tmp_path, kind, name, "proposed", by="kevin")["status"] == "proposed"


def test_the_refusal_to_approve_a_rejected_entry_names_a_way_back_from_the_portal_too(tmp_path):
    _read_again(tmp_path, status="rejected", decided_by="kevin", decided_at="2026-11-04T11:00:00+00:00")
    with pytest.raises(derived.DerivedError, match="Correct in the portal"):
        approve(tmp_path)


@pytest.mark.parametrize("text", ['{"version": 1,', "[1, 2]", '"layer"'])
def test_a_layer_that_is_not_json_or_not_an_object_is_a_derived_error(tmp_path, text):
    (tmp_path / "derived").mkdir()
    derived.path(tmp_path).write_text(text, encoding="utf-8")
    with pytest.raises(derived.DerivedError, match="derived.json: not"):
        derived.read(tmp_path)
