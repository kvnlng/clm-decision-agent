"""Tests for the CrowdStrike scenario and its tools: what the server is told about each passenger."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import FakeRouter
from test_crowdstrike_policy import booking
from test_crowdstrike_world import flight

import crowdstrike_tools
from passengers import Fare
from policy import ROUTES
from world import World

import decision
from scenario import load_scenario
from tools import load_tools

SCENARIO = Path(__file__).parent.parent / "examples" / "crowdstrike" / "scenario.toml"
HIDDEN = {"deadline": 870, "deadline_kind": "firm", "purpose": "a wedding", "still_travelling": False,
          "vulnerable": True, "travel_mode": "rail", "upset": "angry", "vague": True}


@pytest.fixture
def world():
    # Offsets of -240 (Atlanta-like) at A and -360 at B, so local times differ from UTC
    return World([
        flight("X", "A", "B", 600, 700, cancelled=True),
        flight("T1", "A", "B", 800, 900),
        flight("P1", "A", "B", 750, 850, carrier="AS"),
        flight("M1", "A", "B", 1800, 1900),
        # To D, so they are no alternative for the bookings to B
        flight("OK", "A", "D", 600, 700, carrier="UA", actual=(630, 730)),
        flight("LATE", "A", "D", 600, 700, actual=(750, 1000)),
        flight("T2", "A", "D", 800, 900),
        flight("AC", "A", "C", 600, 700, actual=(690, 790)),
        flight("CB", "C", "B", 800, 900),
    ], offsets={"A": -240, "B": -360, "C": -240, "D": -360})


def use(world, *bookings):
    crowdstrike_tools.use(world, {one.id: one for one in bookings})


def facts(one, text="My flight is cancelled.", now=0):
    return crowdstrike_tools.situation({"text": text, "context": {"booking": one.id, "now": now}})["facts"]


def test_route_descriptions_are_the_policy():
    scenario = load_scenario(SCENARIO)
    assert list(scenario.routes) == list(ROUTES)
    assert {name: route.description for name, route in scenario.routes.items()} == ROUTES


def test_scenario_loads_with_its_tools():
    scenario = load_scenario(SCENARIO)
    tools = load_tools(scenario)
    assert {"situation"} <= set(tools)
    assert scenario.route_inform == ("situation",)
    assert set(scenario.questions) == {"deadline", "vulnerable", "travel_mode", "upset"}


def test_cancelled_first_leg_is_stated(world):
    one = booking(fare=Fare(refundable=True, partner_ok=False))
    use(world, one)
    found = facts(one)
    assert found[0] == "The booking, for 1 person: DLX from A at 06:00 to B, due to land 05:40 (local times)."
    assert "DLX is cancelled." in found
    assert "The fare is refundable and does not allow partner airlines." in found
    assert "The soonest of our own flights today with seats for all of them lands at 09:00." in found


def test_on_plan_flights_give_no_alternatives(world):
    one = booking(legs=("OK",), dest="D")
    use(world, one)
    found = facts(one)
    assert "Their flights still go and land within an hour of plan." in found
    assert not any("soonest" in fact or "None of" in fact for fact in found)


def test_a_long_delay_is_stated(world):
    one = booking(legs=("LATE",), dest="D")
    use(world, one)
    found = facts(one)
    assert "Their flights still go but now land at 10:40, 5 h late." in found
    assert "The soonest of our own flights today with seats for all of them lands at 09:00, sooner than " \
           "their flights now will." in found


def test_missed_connection_is_stated(world):
    one = booking(legs=("AC", "CB"))
    use(world, one)
    found = facts(one, now=700)
    assert "DLAC now lands at C at 09:10, too late for the connection to DLCB." in found


def test_partner_and_tomorrow_are_stated_only_when_ours_has_nothing(world):
    full = booking(fare=Fare(refundable=False, partner_ok=True))
    use(world, full)
    world.hold("full-T1", ("T1",), world.free("T1"))
    found = facts(full)
    assert "None of our own flights today with seats for all of them gets them there." in found
    assert "The soonest partner flight today with seats for all of them lands at 08:10." in found
    assert not any("tomorrow" in fact for fact in found)
    world.hold("full-P1", ("P1",), world.free("P1"))
    found = facts(full)
    assert "No partner flight today with seats for all of them gets them there." in found
    assert "Our first flight tomorrow morning with seats for all of them lands at 01:40 on 20 July." in found


def test_deadline_found_and_compared(world):
    one = booking()
    use(world, one)
    found = facts(one, "I must be in B by 08:30, my flight was cancelled")
    assert "The soonest of our own flights today with seats for all of them lands at 09:00, after the 08:30 " \
           "they mention." in found


def test_departure_time_is_not_a_deadline(world):
    one = booking()
    use(world, one)
    assert facts(one, "My 06:00 flight is cancelled") == facts(one, "My flight is cancelled")


def test_deadline_on_the_next_day(world):
    one = booking()
    use(world, one)
    found = facts(one, "I have to be there by 09:00 on 20 July")
    assert any(fact.endswith("before the 09:00 on 20 July they mention.") for fact in found)


def test_no_time_no_deadline_fact(world):
    one = booking()
    use(world, one)
    assert not any("they mention" in fact for fact in facts(one))


def test_tools_ignore_hidden_attributes(world):
    one = booking(fare=Fare(refundable=True, partner_ok=True))
    hidden = replace(one, id="B2", **HIDDEN)
    use(world, one, hidden)
    assert facts(one) == facts(hidden)


def test_now_defaults_to_when_they_wrote(world):
    one = booking(sent=900 * 60)            # after T1 and P1 have left
    use(world, one)
    assert crowdstrike_tools.situation({"text": "", "context": {"booking": one.id}})["facts"] == facts(one, "",
                                                                                                        now=900)


def test_unknown_booking_fails_clearly(world):
    scenario = load_scenario(SCENARIO)
    tools = load_tools(scenario)
    use(world)                              # load_tools configured the real day; take the small world back
    router = FakeRouter({"route": "Agent", "confidence": 0.9, "probabilities": {"Agent": 0.9}, "answers": {},
                         "source": "clm"})
    result = decision.decide("Help", scenario, router, tools, context={"booking": "NOPE"})
    failed = next(entry for entry in result["informed"] if entry["tool"] == "situation")
    assert "context has no known booking: 'NOPE'" in failed["error"]
