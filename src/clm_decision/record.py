"""The record: the owner's corrections of CLM's routes, one JSON object per line beside the scenario."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .scenario import Scenario, ScenarioError, load_for_command


def new_key() -> bytes:
    """A fresh key for encrypting a record."""
    return Fernet.generate_key()


def record_key(scenario: Scenario) -> bytes | None:
    """The key the scenario names for its record, or None if it asks for no encryption."""
    if scenario.record_key_env:
        key = os.environ.get(scenario.record_key_env, "").strip().encode()
        if not key:
            raise ScenarioError(f"record.key_env: the variable {scenario.record_key_env} is not set")
        source = f"the variable {scenario.record_key_env}"
    elif scenario.record_key_file:
        try:
            key = scenario.record_key_file.read_bytes().strip()
        except OSError as error:
            raise ScenarioError(f"record.key_file: cannot read the key file {scenario.record_key_file}: "
                                f"{error.strerror}") from error
        source = str(scenario.record_key_file)
    else:
        return None
    try:
        Fernet(key)
    except ValueError as error:
        raise ScenarioError(f"record: {source} is not a key made by `clm-record newkey`") from error
    return key


def _open_private(path: Path, flags: int) -> int:
    # The record holds request text, so a new file, and a folder made for it, are its owner's only
    Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return os.open(path, flags, 0o600)


def touch_record(path: Path) -> None:
    """Make sure the record can be written, creating it (and its folder) private to its owner."""
    os.close(_open_private(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT))
    if not Path(path).is_symlink():
        os.chmod(path, 0o600)       # also tightens a record made before this protection existed


def append_record(path: Path, entry: dict, key: bytes | None = None) -> None:
    """Append one verdict to the record file, stamped with the current UTC time; encrypted if a key is given."""
    stamped = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    line = json.dumps(stamped, ensure_ascii=False).encode("utf-8")
    if key:
        line = Fernet(key).encrypt(line)
    with os.fdopen(_open_private(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT), "ab") as file:
        file.write(line + b"\n")


def _valid(entry) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("state_text"), str) or not entry["state_text"]:
        return False
    human, clm = entry.get("human"), entry.get("clm")
    return entry.get("kind") == "route" and isinstance(human, str) and isinstance(clm, dict) \
        and isinstance(clm.get("route"), str)


def save_correction(scenario: Scenario, text: str, route: str, clm: dict, more: dict | None = None) -> None:
    """Save the owner's route for `text`, with what CLM had decided ({"route", "confidence"}), and `more`: the
    owner's answers, and the facts and options with the chosen option."""
    if route not in scenario.routes:
        raise ValueError(f"unknown route '{route}'; the routes are {', '.join(scenario.routes)}")
    entry = {"kind": "route", "state_text": text, "human": route,
             "clm": {"route": clm["route"], "confidence": clm["confidence"]}, **(more or {})}
    append_record(scenario.record_path, entry, record_key(scenario))


def _decode(line: bytes, key: bytes | None):
    """One line of the file as a JSON value: plain, or encrypted with `key`. Raises ValueError otherwise."""
    if not line.lstrip().startswith(b"{"):
        if key is None:
            raise ValueError("encrypted line and no key")
        try:
            line = Fernet(key).decrypt(line.strip())
        except InvalidToken as error:
            raise ValueError("not readable with this key") from error
    # Decoded per line so one line cut inside a character cannot hide the rest
    return json.loads(line.decode("utf-8"))


def _lines(path: Path) -> list[bytes]:
    try:
        return [line for line in Path(path).read_bytes().splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def _is_token(line: bytes) -> bool:
    return line.strip().startswith(b"gAAAA")        # how every Fernet token begins


def locked_lines(path: Path, key: bytes | None) -> int:
    """How many lines are encrypted with a key other than this one (or with a key when there is none).

    Writing to such a record would leave a file that no single key can read."""
    locked = 0
    for line in _lines(path):
        if _is_token(line):
            try:
                _decode(line, key)
            except ValueError:
                locked += 1
    return locked


def load_records(path: Path, key: bytes | None = None) -> tuple[list[dict], int]:
    """-> (valid records in file order, number of non-empty lines that could not be used).

    Lines encrypted with another key, or with a key when none is given, count as unusable."""
    records, skipped = [], 0
    for line in _lines(path):
        try:
            entry = _decode(line, key)
        except ValueError:          # also covers bad JSON and bad UTF-8
            skipped += 1
            continue
        if _valid(entry):
            records.append(entry)
        else:
            skipped += 1
    return records, skipped


def route_verdicts(records: list[dict], scenario: Scenario) -> list[dict]:
    """The reviewer's route for each reviewed state, for routes the scenario still has.

    Confirmations count as well as corrections: CLM can pick the right route with too little
    confidence to act on it. Only the latest verdict on a given state is kept.
    """
    latest = {}
    for r in records:
        if r["kind"] == "route" and r["human"] in scenario.routes:
            latest.pop(r["state_text"], None)
            latest[r["state_text"]] = r
    return list(latest.values())


def _newkey(target: Path) -> int:
    if not target.parent.is_dir():
        print(f"error: the folder {target.parent} does not exist.", file=sys.stderr)
        return 2
    try:
        with os.fdopen(_open_private(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "wb") as file:
            file.write(new_key() + b"\n")
    except FileExistsError:
        print(f"error: {target} already exists. A record written with a key cannot be read without it, "
              f"so it is not replaced.", file=sys.stderr)
        return 2
    print(f"Key written to {target}, readable by you only. Keep it out of version control and back it up:\n"
          f"a record written with it cannot be read without it.")
    return 0


def _purge(scenario: Scenario, read) -> int:
    path = scenario.record_path
    if not path.exists() and not path.is_symlink():
        print(f"There is no record at {path}.")
        return 0
    # [record] path can name any file, so only something that is plainly a record is deleted
    if path.is_symlink() or not path.is_file():
        print(f"error: {path} is not an ordinary file, so it was not deleted.", file=sys.stderr)
        return 2
    lines = _lines(path)
    if not all(_is_token(line) or line.lstrip().startswith(b'{"time"') for line in lines):
        print(f"error: {path} does not look like a corrections record, so it was not deleted.", file=sys.stderr)
        return 2
    print(f"{path} holds {len(lines)} line{'' if len(lines) == 1 else 's'} of corrected request text.")
    try:
        answer = read("Delete it? This cannot be undone. [yes/no] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() != "yes":
        print("Nothing was deleted.")
        return 1
    path.unlink()
    print("Deleted. Trained heads and example files, if any, are separate files and were not touched.")
    return 0


def main(argv: list[str] | None = None, read=input) -> int:
    """Command-line entry point: `purge <scenario>` or `newkey <file>`; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Look after a scenario's corrections record.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("purge", help="delete a scenario's record").add_argument("scenario")
    commands.add_parser("newkey", help="write a new encryption key to a file").add_argument("file")
    args = parser.parse_args(argv)
    if args.command == "newkey":
        return _newkey(Path(args.file))
    scenario = load_for_command(args.scenario)
    return 2 if scenario is None else _purge(scenario, read)


if __name__ == "__main__":
    sys.exit(main())
