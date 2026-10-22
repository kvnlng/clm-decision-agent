"""The messages the passengers send, and the answers they imply.

A local model wrote the lines in message_pool.jsonl once (write_pool.py), each for one bucket: what
the passenger wants (to travel, a refund, or they don't say), any time they must or would like to
arrive by, whether someone vulnerable travels with them, how they would like to travel, and how upset
they are. A booking's message is a line from its bucket with the placeholders filled from the
booking, now and then with a typo, in lower case, with an aside, or with a second, unrelated request.
Every line says what its bucket needs (`fits`), so every label can be told from the message.
"""
import json
import random
import re
from dataclasses import asdict
from functools import cache
from pathlib import Path
from string import Formatter

from passengers import Booking
from policy import answer
from world import World

POOL_PATH = Path(__file__).with_name("message_pool.jsonl")
PLACEHOLDERS = {"flight", "city", "dest", "time", "deadline", "purpose", "party", "companion"}
DEADLINES = ["none", "preferred", "firm"]
MODES = ["air", "rail", "either"]
UPSET = ["calm", "annoyed", "angry"]
BUCKETS = ([f"travel|{d}|{v}|{m}|{u}" for d in DEADLINES for v in ("-", "vulnerable") for m in MODES for u in UPSET]
           + [f"{reason}|-|{v}|-|{u}" for reason in ("refund", "vague") for v in ("-", "vulnerable") for u in UPSET])
COMPANIONS = ["my baby daughter", "my 2-year-old", "my elderly mother", "my dad, who uses a wheelchair",
              "my son, who is autistic", "my wife, who is seven months pregnant", "my grandmother, who is 91",
              "my husband, who just had knee surgery"]
PARTY = {1: "just me", 2: "the two of us", 3: "the three of us", 4: "the four of us"}
# Neither may say anything about wanting to travel, a refund, a train or a time to arrive by
ASIDES = ["Sorry, typing this on my phone.", "The queue at the desk here is enormous.", "My phone is at 10%.",
          "Thanks in advance.", "First time this has ever happened to me.", "The app keeps crashing."]
EXTRA = ["Also, where do I collect my checked bag?", "Can I still use the lounge while I wait?",
         "Is there somewhere here to charge my phone?", "Where can I get some food at this hour?"]

REFUND_WORDS = ("refund", "refunded", "money back")
REBOOK_WORDS = ("rebook", "another flight", "next flight", "new flight", "different flight", "later flight",
                "earlier flight", "get me on", "put me on", "book me")
TRAVEL_WORDS = REBOOK_WORDS + ("get there", "get home", "get to", "get us to", "get me to", "make it to", "reach",
                               "still need to", "still want to", "still have to", "next one", "next available",
                               "earliest", "alternative", "catch", "board", "a flight to")
TRAIN_WORDS = ("train", "trains", "rail", "amtrak")
EITHER_WORDS = ("either", "whichever", "don't mind", "do not mind", "or fly", "or a flight", "plane or",
                "flight or", "or plane", "or a plane", "whatever gets")
FIRM_WORDS = ("must", "have to", "has to", "need to be", "got to be", "can't miss", "cannot miss", "can not miss",
              "can't be late", "cannot be late", "can't afford", "absolutely", "essential", "non-negotiable",
              "no later than", "at the latest", "not missing", "no exceptions", "whatever it takes", "by any means")
SOFT_WORDS = ("hope", "hoping", "would like", "i'd like", "we'd like", "ideally", "ideal", "if possible",
              "would be nice", "would be great", "prefer", "rather", "would love", "i'd love", "we'd love", "flexible",
              "okay if we", "fine if we", "okay missing", "not tied", "could miss", "might miss", "don't mind if",
              "not essential", "isn't essential", "can live with", "could live with", "would be a shame")


def _has(lower: str, words: tuple[str, ...]) -> bool:
    return re.search(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", lower) is not None


def placeholders(text: str) -> set[str] | None:
    """The placeholders a line uses, or None when its braces are malformed or carry a format."""
    try:
        parts = list(Formatter().parse(text))
    except ValueError:
        return None
    if any(name is not None and (not name or spec or conversion) for _, name, spec, conversion in parts):
        return None
    return {name for _, name, _, _ in parts if name is not None}


def fits(key: str, text: str) -> bool:  # pylint: disable=too-many-return-statements
    """Whether a line says what its bucket needs, and nothing that belongs to another bucket."""
    reason, deadline, vulnerable, mode, _ = key.split("|")
    names = placeholders(text)
    if names is None or not names <= PLACEHOLDERS:
        return False
    lower = text.lower()
    if _has(lower, REFUND_WORDS) != (reason == "refund"):
        return False
    # A time they must reach their destination by says they still want to go
    if reason == "travel" and not (_has(lower, TRAVEL_WORDS + TRAIN_WORDS) or deadline == "firm"):
        return False
    if reason == "vague" and _has(lower, TRAVEL_WORDS):
        return False
    dated = names & {"deadline", "purpose"}
    if deadline in ("none", "-") and dated:
        return False
    # "Not essential" says the opposite of "essential"
    plain = re.sub(r"\b(?:not|isn't|wasn't|isnt) essential\b", "", lower)
    firm = _has(plain, FIRM_WORDS) or re.search(r"\bneed to (?:get|be)\b[^.]*\{deadline\}", lower)
    if deadline == "firm" and ("deadline" not in names or not firm):
        return False
    if deadline == "preferred" and (not dated or not _has(lower, SOFT_WORDS) or firm):
        return False
    if ("companion" in names) != (vulnerable == "vulnerable"):
        return False
    trains, either = _has(lower, TRAIN_WORDS), _has(lower, EITHER_WORDS)
    return {"air": not trains, "-": not trains, "rail": trains and not either, "either": trains and either}[mode]


def bucket(booking: Booking) -> str:
    """The pool bucket a booking's message comes from."""
    vulnerable = "vulnerable" if booking.vulnerable else "-"
    if booking.vague:
        return f"vague|-|{vulnerable}|-|{booking.upset}"
    if not booking.still_travelling:
        return f"refund|-|{vulnerable}|-|{booking.upset}"
    return f"travel|{booking.deadline_kind}|{vulnerable}|{booking.travel_mode}|{booking.upset}"


class Pool:
    """The message lines, by bucket."""

    def __init__(self, lines: list[dict]):
        self.lines = lines
        self.by_bucket: dict[str, list[str]] = {}
        for line in lines:
            self.by_bucket.setdefault(line["bucket"], []).append(line["text"])

    @classmethod
    def load(cls, path: Path = POOL_PATH) -> "Pool":
        """The pool in a JSON-lines file."""
        with open(path, encoding="utf-8") as file:
            return cls([json.loads(line) for line in file if line.strip()])


@cache
def _committed() -> Pool:
    return Pool.load(POOL_PATH)


def _typo(rng: random.Random, text: str) -> str:
    """Two neighbouring letters swapped in one long word, never inside a placeholder."""
    words = list(re.finditer(r"(?<![{\w])[A-Za-z]{8,}(?![}\w])", text))
    if not words:
        return text
    word = rng.choice(words)
    letters, i = word.group(0), rng.randrange(len(word.group(0)) - 1)
    swapped = letters[:i] + letters[i + 1] + letters[i] + letters[i + 2:]
    return text[:word.start()] + swapped + text[word.end():]


def message(booking: Booking, world: World, seed: int, pool: Pool | None = None) -> str:
    """The message the booking's passenger sends."""
    rng = random.Random(f"{seed}:{booking.id}")
    text = rng.choice((pool or _committed()).by_bucket[bucket(booking)])
    if rng.random() < 0.05:
        text = _typo(rng, text)
    # The model sometimes writes "the {purpose}" or "my {party}", but both carry their own article
    text = re.sub(r"\b(?:[Tt]he|[Aa]n?|[Mm]y|[Oo]ur)\s+(?=\{(?:purpose|party)\})", "", text)
    first = world.flights[booking.legs[0]]
    text = text.format_map({
        "flight": f"{first.carrier}{first.number}", "city": world.cities.get(booking.origin, booking.origin),
        "dest": world.cities.get(booking.dest, booking.dest), "time": world.local(booking.origin, first.sched_dep),
        "deadline": "" if booking.deadline is None else world.local(booking.dest, booking.deadline),
        "purpose": booking.purpose, "party": PARTY[booking.party], "companion": rng.choice(COMPANIONS)})
    if rng.random() < 0.05:
        text = text.lower()
    if rng.random() < 0.10:
        text = f"{text} {rng.choice(ASIDES)}"
    if rng.random() < 0.05:
        text = f"{text} {rng.choice(EXTRA)}"
    return text


def labels(booking: Booking) -> dict:
    """The answers to the scenario's questions, in the examples format, for what the message says."""
    found: dict = {}
    if not booking.vague and booking.still_travelling:
        found["deadline"] = DEADLINES.index(booking.deadline_kind)
        found["travel_mode"] = booking.travel_mode
    found["vulnerable"] = booking.vulnerable
    found["upset"] = UPSET.index(booking.upset)
    return found


def sample(bookings: list[Booking], world: World, seed: int, n: int, pool: Pool | None = None) -> list[dict]:
    """`n` bookings with their message, the policy's answer and the record, for reading through."""
    chosen = sorted(random.Random(f"{seed}:sample").sample(bookings, n), key=lambda b: b.sent)
    lines = []
    for booking in chosen:
        label = answer(booking, world, booking.sent // 60)
        lines.append({"sent": world.local(booking.origin, booking.sent // 60),
                      "text": message(booking, world, seed, pool), "route": label.route,
                      "option": label.option, "answers": labels(booking), "record": asdict(booking)})
    return lines
