"""Tests for the CrowdStrike example's passenger messages."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import json
from collections import Counter
from dataclasses import replace

import pytest
from test_crowdstrike_passengers import small_world

from messages import BUCKETS, COMPANIONS, POOL_PATH, Pool, bucket, fits, labels, message, sample, split
from passengers import make_bookings


def fixture_line(key):
    """A pool line that says what its bucket needs, tagged with the bucket's number."""
    reason, deadline, vulnerable, mode, _ = key.split("|")
    parts = [f"[#{BUCKETS.index(key)}] {{flight}} from {{city}} is off."]
    parts.append({"travel": "Please get me on another flight to {dest}.", "refund": "I want a refund.",
                  "vague": "What happens now?"}[reason])
    parts.append({"firm": "I must be there by {deadline} for {purpose}.", "preferred": "I'd like to make {purpose}.",
                  "none": "", "-": ""}[deadline])
    parts.append("I'm with {companion}." if vulnerable == "vulnerable" else "")
    parts.append({"rail": "I'd rather take the train.", "either": "Train or plane, either works.",
                  "air": "", "-": ""}[mode])
    return " ".join(part for part in parts if part)


@pytest.fixture
def pool():
    return Pool([{"bucket": key, "text": fixture_line(key)} for key in BUCKETS])


@pytest.fixture
def bookings():
    world = small_world()
    return world, make_bookings(world, 0, 1.0)


def test_the_fixture_lines_fit_their_buckets():
    assert all(fits(key, fixture_line(key)) for key in BUCKETS)


@pytest.mark.parametrize("key, text", [
    ("travel|none|-|air|calm", "{flight} is cancelled, please get me on another flight. Or a refund."),
    ("travel|firm|-|air|calm", "Please get me on another flight to {dest}, I have {purpose}."),
    ("travel|preferred|-|air|calm", "Get me on another flight, I must be there by {deadline}."),
    ("travel|none|vulnerable|air|calm", "Please get me on another flight."),
    ("travel|none|-|rail|calm", "Please get me on another flight."),
    ("travel|none|-|air|calm", "Please get me on another flight, or the train."),
    ("travel|none|-|air|calm", "Please get me on another flight to {destination}."),
    ("vague|-|-|-|calm", "What now? I have {purpose}."),
    ("vague|-|-|-|calm", "What now? Can you rebook me?"),
    ("vague|-|-|-|calm", "What now? Is there a train?"),
    ("refund|-|-|-|calm", "I'm done, cancel it."),
])
def test_lines_missing_what_their_bucket_needs_are_discarded(key, text):
    assert not fits(key, text)


def test_saying_an_event_is_not_essential_is_a_preferred_deadline():
    assert fits("travel|preferred|-|air|calm", "Please get me on another flight; {purpose} is not essential.")
    assert not fits("travel|preferred|-|air|calm", "Please get me on another flight; {purpose} is essential.")


def test_every_bucket_is_one_a_booking_can_have(bookings):
    _, made = bookings
    assert {bucket(b) for b in made} <= set(BUCKETS)


def test_messages_are_deterministic_per_seed(bookings, pool):
    world, made = bookings
    first = [message(b, world, 0, pool) for b in made]
    assert first == [message(b, world, 0, pool) for b in made]
    assert first != [message(b, world, 1, pool) for b in made]


def test_every_deciding_attribute_is_expressed(bookings, pool):
    world, made = bookings
    variants = [replace(made[0], still_travelling=s, deadline_kind=k, deadline=None if k == "none" else 900,
                        vulnerable=v, travel_mode=m, vague=False, upset=u)
                for s in (True, False) for k in ("none", "preferred", "firm") for v in (True, False)
                for m in ("air", "rail", "either") for u in ("calm", "annoyed", "angry")]
    for booking in variants + made:
        text = message(booking, world, 0, pool)
        assert f"[#{BUCKETS.index(bucket(booking))}]" in text
        lower = text.lower()
        if booking.vague:
            continue
        if not booking.still_travelling:
            assert "refund" in lower
            continue
        if booking.deadline_kind == "firm":
            assert world.local(booking.dest, booking.deadline).lower() in lower
        assert any(c.lower() in lower for c in COMPANIONS) == booking.vulnerable
        assert ("train" in lower) == (booking.travel_mode != "air")


def test_vague_messages_express_no_deciding_attribute(bookings, pool):
    world, made = bookings
    vague = [replace(b, vague=True, deadline_kind="firm", deadline=900, still_travelling=False) for b in made[:20]]
    for booking in vague:
        lower = message(booking, world, 0, pool).lower()
        assert "refund" not in lower and "another flight" not in lower and booking.purpose not in lower


def test_labels_use_the_examples_format(bookings):
    _, made = bookings
    booking = replace(made[0], vague=False, still_travelling=True, deadline_kind="firm", deadline=900,
                      vulnerable=True, travel_mode="rail", upset="angry")
    assert labels(booking) == {"deadline": 2, "vulnerable": True, "travel_mode": "rail", "upset": 2}
    assert labels(replace(booking, still_travelling=False)) == {"vulnerable": True, "upset": 2}
    assert labels(replace(booking, vague=True)) == {"vulnerable": True, "upset": 2}


def test_a_sample_carries_text_answer_and_record(bookings, pool):
    world, made = bookings
    lines = sample(made, world, 0, 5, pool)
    assert len(lines) == 5 and len({line["record"]["id"] for line in lines}) == 5
    assert set(lines[0]) == {"text", "route", "option", "answers", "record", "sent"}
    json.dumps(lines)


def test_the_pool_has_every_bucket():
    pool = Pool.load(POOL_PATH)
    counts = Counter(line["bucket"] for line in pool.lines)
    assert set(counts) == set(BUCKETS) and min(counts.values()) >= 20
    assert all(fits(line["bucket"], line["text"]) for line in pool.lines)


def test_an_article_before_an_event_or_party_is_dropped(bookings):
    world, made = bookings
    booking = replace(made[0], party=2, deadline_kind="preferred", deadline=900, purpose="a concert", vague=False,
                      still_travelling=True, vulnerable=False, travel_mode="air", upset="calm")
    pool = Pool([{"bucket": bucket(booking), "text": "My {party} and I hope to make the {purpose}."}])
    assert message(booking, world, 0, pool).startswith("the two of us and I hope to make a concert.")


def test_message_does_not_contradict_the_disruption(bookings):
    world, made = bookings
    texts = ["{flight} was cancelled. Rebook me to {dest}.", "{flight} is delayed. Rebook me to {dest}.",
             "{flight} is late. Rebook me to {dest}.", "{flight} let me down. Rebook me to {dest}."]
    one = next(b for b in made if b.still_travelling and not b.vague)
    mixed = Pool([{"bucket": bucket(one), "text": text} for text in texts])
    cancelled = any(world.flights[leg].cancelled for leg in one.legs)
    seen = {message(one, world, seed, mixed) for seed in range(40)}
    wrong = ("delayed", "late") if cancelled else ("cancelled",)
    assert seen and not any(word in text for text in seen for word in wrong)
    # A bucket with only contradicting lines still gives a message
    only = Pool([{"bucket": bucket(one), "text": texts[1 if cancelled else 0]}])
    assert message(one, world, 0, only)


def test_split_is_stable_and_about_a_fifth():
    committed = Pool.load()
    train, unseen = split(committed)
    assert (train.lines, unseen.lines) == (split(committed)[0].lines, split(committed)[1].lines)
    assert len(train.lines) + len(unseen.lines) == len(committed.lines)
    assert 0.15 < len(unseen.lines) / len(committed.lines) < 0.25
    assert not {line["text"] for line in train.lines} & {line["text"] for line in unseen.lines}
