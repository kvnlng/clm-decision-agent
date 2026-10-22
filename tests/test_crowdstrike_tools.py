"""Tests for the CrowdStrike scenario and its tools: what the server is told about each passenger."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import random
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import FakeRouter
from test_crowdstrike_messages import fixture_line
from test_crowdstrike_policy import booking
from test_crowdstrike_world import flight

import crowdstrike_tools
from messages import BUCKETS, Pool
from passengers import Fare
from policy import ROUTES, answer
from world import World
from check import report, score
from write_examples import example

import decision
from dataset import training_rows
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


def request(one, route, now=0, option=None):
    found = {"text": "", "route": route, "context": {"booking": one.id, "now": now}}
    return found if option is None else {**found, "option": {"id": option, "text": ""}}


def router(route, confidence=0.9):
    return FakeRouter({"route": route, "confidence": confidence, "probabilities": {route: confidence},
                       "answers": {}, "source": "clm"})


def test_options_follow_the_route(world):
    one = booking(fare=Fare(refundable=False, partner_ok=True))
    use(world, one)
    ids = {route: [option["id"] for option in crowdstrike_tools.alternatives(request(one, route))["options"]]
           for route in ("Rebook", "Rebook_Partner", "Overnight")}
    assert ids == {"Rebook": ["T1"], "Rebook_Partner": ["P1"], "Overnight": ["M1"]}
    first = crowdstrike_tools.alternatives(request(one, "Rebook"))["options"][0]["text"]
    assert first == "DLT1 from A at 09:20 to B, arriving 09:00 (local times); lands first of these"


def test_label_option_is_the_first_alternative():
    crowdstrike_tools.configure({"seed": 0}, SCENARIO.parent)
    day, bookings = crowdstrike_tools.STATE["world"], list(crowdstrike_tools.STATE["bookings"].values())
    seen = set()
    for one in random.Random(1).sample(bookings, 2000):
        label = answer(one, day, one.sent // 60)
        if label.route in ("Rebook", "Rebook_Partner", "Overnight"):
            seen.add(label.route)
            options = crowdstrike_tools.alternatives(request(one, label.route, one.sent // 60))["options"]
            assert options[0]["id"] == "+".join(label.option)
    assert seen == {"Rebook", "Rebook_Partner", "Overnight"}


def test_hold_that_loses_the_race_fails(world):
    one = booking()
    scenario = load_scenario(SCENARIO)
    tools = load_tools(scenario)
    use(world, one)

    def alternatives(found):
        options = crowdstrike_tools.alternatives(found)
        world.hold("someone-else", ("T1",), world.free("T1"))      # between choosing and holding
        return options

    result = decision.decide("Rebook me", scenario, router("Rebook"), {**tools, "alternatives": alternatives},
                             context={"booking": one.id, "now": 0})
    held = next(entry for entry in result["actions"] if entry["tool"] == "hold_seats")
    assert held["status"] == "failed" and "T1 has 0 seats left" in held["error"]


def test_a_hold_takes_the_partys_seats(world):
    one = booking(party=3)
    use(world, one)
    before = world.free("T1")
    assert crowdstrike_tools.hold_seats(request(one, "Rebook", option="T1")) == {"held": ["T1"]}
    assert world.free("T1") == before - 3


def test_uncertain_route_holds_nothing(world):
    one = booking()
    scenario = load_scenario(SCENARIO)
    tools = load_tools(scenario)
    use(world, one)
    before = world.free("T1")
    result = decision.decide("Rebook me", scenario, router("Rebook", 0.3), tools,
                             context={"booking": one.id, "now": 0})
    assert {entry["status"] for entry in result["actions"]} == {"skipped"}
    assert world.free("T1") == before


def test_hotel_only_on_overnight():
    scenario = load_scenario(SCENARIO)
    routes = {action.tool: set(action.routes) for action in scenario.actions}
    assert routes == {"hold_seats": {"Rebook", "Rebook_Partner", "Overnight"}, "book_hotel": {"Overnight"},
                      "issue_refund": {"Refund"},
                      "draft_reply": {"Rebook", "Rebook_Partner", "Overnight", "Refund", "No_Action"}}


def test_the_reply_names_the_journey(world):
    one = booking()
    use(world, one)
    crowdstrike_tools.hold_seats(request(one, "Overnight", option="M1"))
    reply = crowdstrike_tools.draft_reply(request(one, "Overnight", option="M1"))["reply"]
    assert "a hotel for tonight" in reply and "DLM1" in reply


def test_no_hotel_or_booked_reply_when_the_hold_failed(world):
    # The acting tools run in order whatever happened before: hold_seats failed, so nothing is held
    one = booking()
    use(world, one)
    with pytest.raises(ValueError, match="no seats are held"):
        crowdstrike_tools.book_hotel(request(one, "Overnight", option="M1"))
    with pytest.raises(ValueError, match="no seats are held"):
        crowdstrike_tools.draft_reply(request(one, "Overnight", option="M1"))


def test_example_facts_match_the_server(world):
    one = booking(fare=Fare(refundable=False, partner_ok=True))
    use(world, one)
    pool = Pool([{"bucket": key, "text": fixture_line(key)} for key in BUCKETS])
    line = example(one, world, pool, 0)
    assert (line["route"], line["option"]) == ("Rebook", "T1")
    scenario = load_scenario(SCENARIO)
    tools = load_tools(scenario)
    use(world, one)
    result = decision.decide(line["text"], scenario, router("Rebook"), tools, context=line["context"])
    informed = {entry["tool"]: entry for entry in result["informed"]}
    assert line["facts"] == {"situation": informed["situation"]["facts"]}
    assert line["options"] == informed["alternatives"]["options"]


def test_examples_load_in_train(world):
    made = [booking(id=f"B{i}", legs=(legs,), dest=dest) for i, (legs, dest) in enumerate([("X", "B"), ("LATE", "D"),
                                                                                         ("OK", "D")])]
    use(world, *made)
    pool = Pool([{"bucket": key, "text": fixture_line(key)} for key in BUCKETS])
    lines = [example(one, world, pool, 0) for one in made]
    train, _, test = training_rows(load_scenario(SCENARIO), lines, [])
    rows = train + test
    assert len(rows) == 3 and all("\n\nKnown:\n- The booking" in row["text"] for row in rows)
    assert {row["route"] for row in rows} == {"Rebook", "No_Action"}


def test_check_reports_per_class():
    cases = [{"route": "Rebook", "option": "T1", "answers": {"deadline": 2, "vulnerable": True, "travel_mode": "air"}},
             {"route": "Rebook", "option": "T1", "answers": {"deadline": 0, "vulnerable": False}},
             {"route": "Refund", "answers": {"upset": 2}}]
    replies = [{"route": "Rebook", "uncertain": False, "option": {"id": "T1"},
                "answers": {"deadline": {"type": "score", "score": 1.6}, "vulnerable": {"type": "noul", "noul": 0.7},
                            "travel_mode": {"type": "choice", "choice": "rail"}}},
               {"route": "Rebook", "uncertain": False, "option": {"id": "T2"},
                "answers": {"deadline": {"type": "score", "score": 0.4}, "vulnerable": {"type": "noul", "noul": 0.6}}},
               {"route": "Agent", "uncertain": True, "answers": {"upset": {"type": "score", "score": 1.2}}}]
    found = score(cases, replies)
    assert found["route"] == {"Rebook": (2, 2), "Refund": (0, 1)}
    assert found["option"] == {"Rebook": (1, 2)}
    assert found["wrong_uncertain"] == (1, 1)
    assert found["answers"]["deadline"] == {0: (1, 1), 2: (1, 1)}
    assert found["answers"]["vulnerable"] == {True: (1, 1), False: (0, 1)}
    assert found["answers"]["travel_mode"] == {"air": (0, 1)}
    assert found["answers"]["upset"] == {2: (0, 1)}
    assert "Rebook" in report(found) and "2 of 2" in report(found)


def test_a_time_ours_misses_brings_the_partner_and_tomorrow(world):
    # Our T1 lands 09:00, after the 08:30 mentioned, so the partner's P1 (08:10) is what the label turns on
    one = booking(fare=Fare(refundable=False, partner_ok=True))
    use(world, one)
    found = facts(one, "I must be in B by 08:30")
    assert "The soonest partner flight today with seats for all of them lands at 08:10, before the 08:30 they " \
           "mention." in found
    assert not any("tomorrow" in fact for fact in found)
    # Without a partner, tomorrow is stated too
    use(world, replace(one, fare=Fare(refundable=False, partner_ok=False)))
    assert any("tomorrow morning" in fact for fact in facts(one, "I must be in B by 08:30"))
