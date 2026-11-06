"""Tests for holder.py: each call keeps the environment it started with, and one swapped out is closed when its last
call ends, keeping open what a live environment shares with it."""
# Test names say what they check.
# pylint: disable=missing-function-docstring
from dataclasses import replace

from conftest import ENV_TOML
from fake_documents import FakeEmbedder, FakeStore

from clm_decision.environment import load_environment
from clm_decision.holder import Holder


class ClosableStore(FakeStore):
    """A fake document store that counts how often it was closed."""

    def __init__(self):
        super().__init__()
        self.closed = 0

    def close(self):
        self.closed += 1


WITH_DOCUMENTS = ENV_TOML + '\n[documents]\nbackend = "postgres"\npassword_env = "TOY_DOCUMENTS_PASSWORD"\n'


def environment(write_environment):
    return load_environment(write_environment(WITH_DOCUMENTS), store=ClosableStore(), embedder=FakeEmbedder())


def test_use_gives_the_current_environment(write_environment):
    env = environment(write_environment)
    holder = Holder(env)
    with holder.use() as used:
        assert used is env and holder.current is env


def test_swap_with_no_call_closes_the_old_one_at_once(write_environment):
    old = environment(write_environment)
    new = replace(old, store=ClosableStore())
    holder = Holder(old)
    assert holder.swap(new) is old and holder.current is new
    assert old.store.closed == 1 and new.store.closed == 0


def test_a_call_in_flight_keeps_the_old_environment_open_until_it_ends(write_environment):
    old = environment(write_environment)
    new = replace(old, store=ClosableStore())
    holder = Holder(old)
    with holder.use() as first:
        with holder.use() as second:
            holder.swap(new)
            with holder.use() as after:
                assert after is new
            assert first is second is old and old.store.closed == 0
        assert old.store.closed == 0
    assert old.store.closed == 1 and new.store.closed == 0


def test_swapping_in_the_current_environment_changes_nothing(write_environment):
    env = environment(write_environment)
    holder = Holder(env)
    assert holder.swap(env) is env and holder.current is env and env.store.closed == 0


def test_a_store_shared_by_an_environment_still_in_use_is_not_closed(write_environment):
    # a holds a call; b shares a's store; c has its own. b is retired with no call: closing it must not close the
    # store a's call is still reading
    a = environment(write_environment)
    b = replace(a)
    c = replace(a, store=ClosableStore())
    holder = Holder(a)
    with holder.use():
        holder.swap(b)
        holder.swap(c)
        assert a.store.closed == 0
    assert a.store.closed == 1 and c.store.closed == 0


def test_a_shared_store_is_never_closed_while_the_current_environment_holds_it(write_environment):
    old = environment(write_environment)
    new = replace(old)
    holder = Holder(old)
    with holder.use():
        holder.swap(new)
    assert old.store.closed == 0
