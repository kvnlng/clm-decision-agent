"""The derived layer: rules and facts derived from an environment's documents, kept apart from the supplied ones.

It lives in `derived/derived.json` beside the environment file, with any decision tables it derived in
`derived/rules/`. Each entry has a status: proposed when written, checked once it passed the mechanical check,
then approved or rejected by a person; a conflict with a supplied fact is open until a person resolves it. Only
this module changes a status, and every change records who made it and when: set_status a person's decision, with
why; set_check the mechanical check's result, and set_stale an entry whose cited document is no longer
the edition in force, both recorded as made by clm-derive check.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

VERSION = 1
KINDS = ("decisions", "facts", "conflicts")
STATUSES = {"decisions": ("proposed", "checked", "approved", "rejected"),
            "facts": ("proposed", "checked", "approved", "rejected"),
            "conflicts": ("open", "resolved")}


class DerivedError(ValueError):
    """A derived layer that cannot be read, or a status change that is not allowed."""


def path(folder: str | Path) -> Path:
    """Where an environment folder's derived layer is kept."""
    return Path(folder) / "derived" / "derived.json"


def empty(collection: str | None) -> dict:
    """A layer with nothing derived yet, from the named collection."""
    return {"version": VERSION, "collection": collection, "decisions": {}, "facts": {}, "conflicts": {}}


def read(folder: str | Path) -> dict:
    """An environment folder's derived layer, or an empty one when nothing has been derived."""
    file = path(folder)
    if not file.exists():
        return empty(None)
    layer = json.loads(file.read_text(encoding="utf-8"))
    if layer.get("version") != VERSION or any(not isinstance(layer.get(kind), dict) for kind in KINDS):
        raise DerivedError(f"{file}: not a derived layer of version {VERSION}")
    return layer


def write(folder: str | Path, layer: dict) -> None:
    """Write the layer, replacing the file whole so a reader never sees half of it."""
    file = path(folder)
    file.parent.mkdir(parents=True, exist_ok=True)
    partial = file.with_suffix(".json.partial")
    partial.write_text(json.dumps(layer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, file)


def dmn_digest(folder: str | Path, entry: dict) -> str | None:
    """The sha256 of the entry's decision table as it is now; None for an entry with no table, or one not there."""
    if not entry.get("dmn"):
        return None
    try:
        return hashlib.sha256((Path(folder) / "derived" / entry["dmn"]).read_bytes()).hexdigest()
    except OSError:
        return None


def changed_since_check(folder: str | Path, entry: dict) -> bool:
    """Whether the entry's table is no longer the one its check passed. A check recorded before digests were kept
    (no dmn_sha256) is taken as it is."""
    recorded = (entry.get("check") or {}).get("dmn_sha256")
    return recorded is not None and recorded != dmn_digest(folder, entry)


def changed_line(name: str) -> str:
    """What to say of an entry whose table changed since its check."""
    return f"{name}'s DMN has changed since its check: run clm-derive check again"


# pylint: disable-next=too-many-arguments
def set_status(folder: str | Path, kind: str, name: str, status: str, *, by: str, note: str | None = None) -> dict:
    """Change one entry's status, recording who did it, when and why, and return the entry. A rule or fact is
    approved only once checked: a proposed one, such as a corrected one, is not, whatever its old check said. Any
    can be rejected."""
    if kind not in KINDS:
        raise DerivedError(f"{kind!r} is not one of {', '.join(KINDS)}")
    if status not in STATUSES[kind]:
        raise DerivedError(f"{status!r} is not a status of {kind}: {', '.join(STATUSES[kind])}")
    if not by:
        raise DerivedError("a status change needs by: who made it")
    layer = read(folder)
    entry = layer[kind].get(name)
    if entry is None:
        raise DerivedError(f"no {kind} entry named {name!r}"
                           + (": it is not derived: run clm-derive run" if kind != "conflicts" else ""))
    if status == "approved" and not (entry.get("check") or {}).get("passed"):
        raise DerivedError(f"{name} failed its check, or was never checked, so it cannot be approved")
    if status == "approved" and entry.get("status") == "proposed":
        raise DerivedError(f"{name} is proposed: it is approved only once clm-derive check has checked it")
    if status == "approved" and changed_since_check(folder, entry):
        raise DerivedError(changed_line(name))
    entry.update(status=status, decided_by=by, note=note,
                 decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    write(folder, layer)
    return entry


CHECKER = "clm-derive check"


def set_check(folder: str | Path, kind: str, name: str, check: dict) -> dict:
    """Record the mechanical check's result on a proposed or checked entry, with when and by clm-derive check, and
    return the entry: one that passed becomes checked, one that failed proposed. A second reading's record
    ({model, agreed, differences}) is kept when one was made, and the sha256 of the table checked when there is
    one. An approved or rejected entry is a person's decision and is never changed here, nor are the fields
    recording it."""
    if kind not in ("decisions", "facts"):
        raise DerivedError(f"{kind!r}: only decisions or facts are checked")
    layer = read(folder)
    entry = layer[kind].get(name)
    if entry is None:
        raise DerivedError(f"no {kind} entry named {name!r}")
    if entry.get("status") not in ("proposed", "checked"):
        raise DerivedError(f"{name} is {entry.get('status')}: only proposed or checked entries are checked")
    passed, digest = check.get("passed") is True, dmn_digest(folder, entry)
    entry.update(status="checked" if passed else "proposed",
                 check={"passed": passed, "failures": list(check.get("failures") or []),
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "by": CHECKER,
                        **({"dmn_sha256": digest} if digest else {}),
                        **({"second_reading": check["second_reading"]} if check.get("second_reading") else {})})
    write(folder, layer)
    return entry


def set_stale(folder: str | Path, kind: str, name: str, failure: str) -> dict:
    """Send an entry whose cited document is no longer the edition in force back to proposed, whatever its status
    but rejected, recording the failure as clm-derive check's, and return the entry. Who decided and when are kept,
    and the note gains the reason, so the record shows the approval was for the earlier edition."""
    if kind not in ("decisions", "facts"):
        raise DerivedError(f"{kind!r}: only decisions or facts are checked")
    layer = read(folder)
    entry = layer[kind].get(name)
    if entry is None:
        raise DerivedError(f"no {kind} entry named {name!r}")
    if entry.get("status") == "rejected":
        raise DerivedError(f"{name} is rejected: a person's rejection stands")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    note = f"{CHECKER}, {now}: back to proposed from {entry.get('status')}: {failure}"
    entry.update(status="proposed", note=f"{entry['note']}; {note}" if entry.get("note") else note,
                 check={"passed": False, "failures": [failure], "at": now, "by": CHECKER})
    write(folder, layer)
    return entry
