"""Tests for record.py."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import json
import stat

import pytest
from conftest import BASE_TOML

from record import append_record, load_records, locked_lines, new_key, record_key, route_verdicts
from record import main as record_main
from scenario import ScenarioError, load_scenario


def route_record(state_text, clm, human):
    return {"kind": "route", "state_text": state_text, "human": human,
            "clm": {"route": clm, "confidence": 0.8, "probabilities": {clm: 0.9}}}


def test_append_then_load_round_trip(tmp_path):
    path = tmp_path / "scenario.record.jsonl"
    append_record(path, route_record("User: hi", "Billing", "Tech"))
    append_record(path, {"kind": "tool", "state_text": "User: x", "calls": [],
                         "clm": {"p": 0.9, "approved": True}, "human": False})
    records, skipped = load_records(path)

    assert skipped == 0
    assert [r["kind"] for r in records] == ["route", "tool"]
    assert records[0]["human"] == "Tech"
    assert records[0]["time"].endswith("+00:00")
    assert len(path.read_text().splitlines()) == 2


def test_missing_file_is_an_empty_record(tmp_path):
    assert load_records(tmp_path / "none.jsonl") == ([], 0)


def test_damaged_lines_are_skipped_and_counted(tmp_path):
    path = tmp_path / "r.jsonl"
    good = json.dumps(route_record("User: hi", "Billing", "Tech"))
    path.write_text("\n".join([
        good,
        '{"kind": "route", "state_text": "User: cut o',        # write interrupted
        '["not", "an", "object"]',
        '{"kind": "mystery", "human": true}',                    # unknown kind
        '{"kind": "route", "human": "Tech"}',                    # missing state_text and clm
        "",
        good,
    ]) + "\n")
    records, skipped = load_records(path)
    assert len(records) == 2
    assert skipped == 4


def test_route_verdicts_are_the_reviewed_routes_the_scenario_still_has(scenario):
    records = [
        route_record("User: a", "Billing", "Tech"),      # correction
        route_record("User: b", "Billing", "Billing"),   # confirmation
        route_record("User: c", "Tech", "Shipping"),     # route no longer in the scenario
        {"kind": "tool", "state_text": "User: d", "calls": [], "clm": {"p": 0.9, "approved": True}, "human": False},
    ]
    assert [r["state_text"] for r in route_verdicts(records, scenario)] == ["User: a", "User: b"]


def test_line_cut_inside_a_multibyte_character_is_skipped(tmp_path):
    path = tmp_path / "r.jsonl"
    good = json.dumps(route_record("User: café", "Billing", "Tech"), ensure_ascii=False).encode()
    cut = good[: good.index("é".encode()) + 1]
    path.write_bytes(good + b"\n" + cut + b"\n" + good + b"\n")
    records, skipped = load_records(path)
    assert (len(records), skipped) == (2, 1)


def test_a_later_confirmation_replaces_an_earlier_correction(scenario):
    records = [route_record("User: a", "Billing", "Tech"), route_record("User: a", "Billing", "Billing")]
    assert [r["human"] for r in route_verdicts(records, scenario)] == ["Billing"]


def test_a_later_correction_replaces_an_earlier_verdict(scenario):
    records = [route_record("User: a", "Billing", "Billing"), route_record("User: a", "Billing", "Tech")]
    assert [r["human"] for r in route_verdicts(records, scenario)] == ["Tech"]


def test_the_record_is_created_readable_by_its_owner_only(tmp_path):
    path = tmp_path / "r.jsonl"
    append_record(path, route_record("User: hi", "Billing", "Tech"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_an_encrypted_record_round_trips_and_holds_no_readable_text(tmp_path):
    path, key = tmp_path / "r.jsonl", new_key()
    append_record(path, route_record("User: my secret question", "Billing", "Tech"), key=key)
    assert b"secret" not in path.read_bytes() and b"Billing" not in path.read_bytes()
    records, skipped = load_records(path, key=key)
    assert (records[0]["state_text"], skipped) == ("User: my secret question", 0)


def test_encrypted_lines_are_skipped_without_the_right_key(tmp_path):
    path = tmp_path / "r.jsonl"
    append_record(path, route_record("User: hi", "Billing", "Tech"), key=new_key())
    assert load_records(path) == ([], 1)
    assert load_records(path, key=new_key()) == ([], 1)


def test_plain_lines_written_before_a_key_was_set_are_still_read(tmp_path):
    path, key = tmp_path / "r.jsonl", new_key()
    append_record(path, route_record("User: old", "Billing", "Tech"))
    append_record(path, route_record("User: new", "Billing", "Tech"), key=key)
    records, skipped = load_records(path, key=key)
    assert ([r["state_text"] for r in records], skipped) == (["User: old", "User: new"], 0)


def with_record(write_scenario, lines):
    return load_scenario(write_scenario(BASE_TOML + "\n[record]\n" + lines))


def test_record_key_comes_from_the_named_variable_or_file(write_scenario, tmp_path, monkeypatch):
    key = new_key()
    assert record_key(load_scenario(write_scenario())) is None

    monkeypatch.setenv("DEMO_RECORD_KEY", key.decode())
    assert record_key(with_record(write_scenario, 'key_env = "DEMO_RECORD_KEY"\n')) == key

    (tmp_path / "record.key").write_bytes(key + b"\n")
    assert record_key(with_record(write_scenario, 'key_file = "record.key"\n')) == key


@pytest.mark.parametrize("lines, message", [
    ('key_env = "DEMO_MISSING_KEY"\n', "the variable DEMO_MISSING_KEY is not set"),
    ('key_file = "nowhere.key"\n', "cannot read the key file"),
    ('key_env = "DEMO_BAD_KEY"\n', "is not a key made by"),
])
def test_a_key_that_is_asked_for_but_unusable_is_an_error(write_scenario, monkeypatch, lines, message):
    monkeypatch.delenv("DEMO_MISSING_KEY", raising=False)
    monkeypatch.setenv("DEMO_BAD_KEY", "not-a-key")
    with pytest.raises(ScenarioError, match=message):
        record_key(with_record(write_scenario, lines))


def test_newkey_writes_a_private_key_file_and_will_not_replace_one(tmp_path, capsys):
    target = tmp_path / "record.key"
    assert record_main(["newkey", str(target)]) == 0
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    first = target.read_bytes()
    append_record(tmp_path / "r.jsonl", route_record("User: hi", "Billing", "Tech"), key=first.strip())
    assert record_main(["newkey", str(target)]) == 2
    assert target.read_bytes() == first


def test_purge_deletes_the_record_only_after_a_yes(write_scenario, capsys):
    scenario = load_scenario(write_scenario())
    append_record(scenario.record_path, route_record("User: hi", "Billing", "Tech"))
    assert record_main(["purge", str(scenario.path)], read=lambda prompt: "n") == 1
    assert scenario.record_path.exists()
    assert record_main(["purge", str(scenario.path)], read=lambda prompt: "yes") == 0
    assert not scenario.record_path.exists()
    assert record_main(["purge", str(scenario.path)], read=lambda prompt: "yes") == 0
    assert "1 line" in capsys.readouterr().out


def test_lines_under_another_key_are_counted_separately_from_damage(tmp_path):
    path, key = tmp_path / "r.jsonl", new_key()
    append_record(path, route_record("User: a", "Billing", "Tech"), key=new_key())
    append_record(path, route_record("User: b", "Billing", "Tech"), key=key)
    with open(path, "ab") as file:
        file.write(b'{"kind": "route"}\n{not json\n')
    assert locked_lines(path, key) == 1
    assert locked_lines(path, None) == 2
    assert locked_lines(tmp_path / "none.jsonl", key) == 0


def test_purge_refuses_anything_that_is_not_a_record_file(write_scenario, tmp_path, capsys):
    victim = tmp_path / "notes.txt"
    victim.write_text("my notes\nmore notes\n")
    for setting in ('path = "notes.txt"', 'path = "."', 'path = "link.jsonl"'):
        scenario = with_record(write_scenario, setting + "\n")
        if "link" in setting:
            real = tmp_path / "real.jsonl"
            append_record(real, route_record("User: hi", "Billing", "Tech"))
            (tmp_path / "link.jsonl").symlink_to(real)
        assert record_main(["purge", str(scenario.path)], read=lambda prompt: "yes") == 2
    assert victim.exists() and (tmp_path / "real.jsonl").exists() and (tmp_path / "link.jsonl").is_symlink()
    assert "not deleted" in capsys.readouterr().err


def test_purge_treats_end_of_input_as_no(write_scenario):
    scenario = load_scenario(write_scenario())
    append_record(scenario.record_path, route_record("User: hi", "Billing", "Tech"))

    def closed(prompt):
        raise EOFError
    assert record_main(["purge", str(scenario.path)], read=closed) == 1
    assert scenario.record_path.exists()


def test_a_new_record_folder_is_private_and_newkey_explains_a_missing_folder(write_scenario, tmp_path, capsys):
    scenario = with_record(write_scenario, 'path = "private/desk.jsonl"\n')
    append_record(scenario.record_path, route_record("User: hi", "Billing", "Tech"))
    assert stat.S_IMODE((tmp_path / "private").stat().st_mode) == 0o700
    assert record_main(["newkey", str(tmp_path / "missing" / "record.key")]) == 2
    assert "does not exist" in capsys.readouterr().err
