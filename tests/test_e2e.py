"""End-to-end tests with the real models. Run with: uv run pytest -m slow"""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# Fakes mirror real interfaces (so some arguments go unused) and note what they were called with.
# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name
# pylint: disable=unused-argument,attribute-defined-outside-init,use-implicit-booleaness-not-comparison
import shutil
from pathlib import Path

import pytest
import service_desk_tools
from conftest import FakeRouter
from desk_routing_cases import CASES
from langchain_core.messages import HumanMessage, ToolMessage

from chat import new_session, run_turn
from clm_router import LocalContrastiveRouter
from generator import make_generator
from graph import build_graph
from scenario import load_scenario
from tools import load_catalogue, load_tools


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def shipped():
    return load_scenario(Path(__file__).parent / "data" / "support_scenario.toml")


@pytest.fixture(scope="module")
def real_router(shipped):
    return LocalContrastiveRouter(shipped.clm_model_id)


@pytest.fixture(scope="module")
def real_generator(shipped):
    return make_generator(shipped)


def test_turn_returns_a_reply_and_a_decision(shipped, real_router, real_generator):
    graph = build_graph(shipped, real_router, real_generator)
    out = graph.invoke({"messages": [HumanMessage(
        content="My invoice was charged twice and nobody answers the phone!")]})
    assert out["decision"]["route"] in shipped.routes
    assert 0.0 <= out["decision"]["confidence"] <= 1.0
    assert str(out["messages"][-1].content).strip()


def test_real_generator_calls_lookup_invoice(shipped, real_generator):
    # Routing is fixed so the test exercises the generator and the tool loop, not CLM's choice
    forced = {"route": "Refund_Processor", "confidence": 0.9, "answers": {}, "source": "clm",
              "probabilities": {"Refund_Processor": 0.9, "Tech_Support": 0.1}}
    graph = build_graph(shipped, FakeRouter(forced, tool_p=0.9), real_generator)
    out = graph.invoke({"messages": [HumanMessage(
        content="Invoice INV-2041 was charged twice, can you check it?")]})

    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages and tool_messages[0].name == "lookup_invoice"
    assert "INV-2041" in tool_messages[0].content
    assert str(out["messages"][-1].content).strip()


def test_corrected_case_is_matched_when_sent_again(shipped, real_router):
    text = "User: I'd like to talk to someone about my bill"
    clm = real_router.decide(text, shipped)
    other = next(name for name in shipped.routes if name != clm["route"])
    records = [{"kind": "route", "state_text": text, "human": other,
                "clm": {"route": clm["route"], "confidence": clm["confidence"],
                        "probabilities": clm["probabilities"]}}]
    try:
        assert real_router.load_verdicts(records, shipped) == 1
        again = real_router.decide(text, shipped)
        assert (again["route"], again["source"], again["clm_route"]) == (other, "reviewed", clm["route"])
        assert again["similarity"] > 0.99
    finally:
        real_router.verdicts = []


def test_reply_after_a_declined_tool_call_does_not_invent_invoice_details(shipped, real_generator):
    forced = {"route": "Refund_Processor", "confidence": 0.9, "answers": {}, "source": "clm",
              "probabilities": {"Refund_Processor": 0.9, "Tech_Support": 0.1}}
    graph = build_graph(shipped, FakeRouter(forced, tool_p=0.1), real_generator)
    out = graph.invoke({"messages": [HumanMessage(
        content="Invoice INV-2041 was charged twice, can you check it?")]})

    assert out["tool_checks"] and not any(check["approved"] for check in out["tool_checks"])
    reply = str(out["messages"][-1].content)
    assert reply.strip()
    assert "$" not in reply


def test_correction_matches_a_paraphrase_but_not_an_unrelated_message(shipped, real_router):
    text = "User: I'd like to talk to someone about my bill"
    clm = real_router.decide(text, shipped)
    other = next(name for name in shipped.routes if name != clm["route"])
    records = [{"kind": "route", "state_text": text, "human": other,
                "clm": {"route": clm["route"], "confidence": clm["confidence"],
                        "probabilities": clm["probabilities"]}}]
    try:
        real_router.load_verdicts(records, shipped)
        paraphrase = real_router.decide("User: I would like to speak with somebody about my bill", shipped)
        unrelated = real_router.decide("User: The app crashes every time I open the settings page", shipped)
        assert paraphrase["source"] == "reviewed"
        assert unrelated["source"] == "clm"
    finally:
        real_router.verdicts = []


def test_question_repeated_later_in_a_conversation_meets_its_correction(shipped, real_router):
    first = "User: My invoice INV-2041 was charged twice"
    clm = real_router.decide(first, shipped)
    other = next(name for name in shipped.routes if name != clm["route"])
    records = [{"kind": "route", "state_text": first, "human": other,
                "clm": {"route": clm["route"], "confidence": clm["confidence"],
                        "probabilities": clm["probabilities"]}}]
    later = first + "\nAssistant: I can look into that for you.\nUser: My invoice INV-2041 was charged twice"
    try:
        real_router.load_verdicts(records, shipped)
        again = real_router.decide(later, shipped)
        assert (again["route"], again["source"]) == (other, "reviewed")
    finally:
        real_router.verdicts = []


DESK_KEYS = ("messages", "draft", "draft_route", "draft_start", "follow_up")


@pytest.fixture(scope="module")
def desk(tmp_path_factory):
    # A copy, so a test cannot change the example's files
    example = Path(__file__).parent.parent / "examples" / "service_desk"
    folder = tmp_path_factory.mktemp("desk")
    for name in ("scenario.toml", "queues.json"):
        shutil.copy(example / name, folder / name)
    return load_scenario(folder / "scenario.toml")


@pytest.fixture(scope="module")
def desk_generator(desk):
    return make_generator(desk)


def forced(route):
    """Routing is fixed so these tests exercise the generator and the graph, not CLM's choice."""
    return FakeRouter({"route": route, "confidence": 0.9, "answers": {}, "source": "clm",
                       "probabilities": {route: 0.9, "Out_Of_Scope": 0.1}}, tool_p=0.9)


def desk_turn(graph, session, text):
    out = run_turn(graph, session, text, ask=lambda payload: {"approved": True})
    return out, {key: out.get(key) for key in DESK_KEYS}


def test_extraction_fills_only_what_the_user_stated(desk, desk_generator):
    graph = build_graph(desk, forced("Data_Access"), desk_generator)
    out, session = desk_turn(graph, new_session(), "I need to get into the readmissions data")
    assert "dataset" in out["draft"] and "readmission" in out["draft"]["dataset"].lower()
    assert "purpose" in out["intake"]["missing"] and "duration" in out["intake"]["missing"]
    assert str(out["messages"][-1].content).strip()

    out, _ = desk_turn(graph, session, "Not sure what it's for yet, my manager asked me to get it. Three months.")
    assert "purpose" not in out["draft"]
    assert "duration" in out["draft"]


def test_a_conversation_ends_with_a_ticket_in_the_tracker(desk, desk_generator, tracker_server, tracker_url):
    registry = load_tools(desk)
    service_desk_tools.SETTINGS["tracker_url"] = tracker_url
    before = set(tracker_server.tracker.issues)
    graph = build_graph(desk, forced("Data_Access"), desk_generator, registry=registry)

    out, session = desk_turn(graph, new_session(),
                             "Can I have access to the oncology registry for six months? I'm reviewing "
                             "treatment delays for our department and de-identified records are enough.")
    for _ in range(2):
        if set(tracker_server.tracker.issues) - before:
            break
        out, session = desk_turn(graph, session, "Yes, please submit it.")

    created = set(tracker_server.tracker.issues) - before
    assert len(created) == 1
    number = created.pop()
    issue = tracker_server.tracker.issues[number]
    assert issue["title"].strip() and "oncology" in (issue["title"] + issue["body"]).lower()
    assert session["draft"] == {} and session["draft_route"] is None
    assert str(number) in str(out["messages"][-1].content)


# Measured when this test was written; CLM's untrained routing is weak, so this only guards
# against the scenario's wording collapsing to one route. Part 2 (training) raises it.
ROUTING_FLOOR = 5


def test_demo_routing_stays_above_its_floor(desk):
    router = LocalContrastiveRouter(desk.clm_model_id)
    results = [(router.decide(f"User: {message}", desk)["route"], route) for message, route in CASES]
    correct = sum(got == want for got, want in results)
    print(f"demo routing: {correct} of {len(CASES)} correct; chosen: "
          f"{ {name: sum(got == name for got, _ in results) for name in desk.routes} }")
    assert correct >= ROUTING_FLOOR, f"{correct} of {len(CASES)} routed correctly"


PICKS = [("I need the cancer registry data", "Oncology Registry"),
         ("Can I see how long people wait in A&E?", "Emergency Department Attendances"),
         ("I want the theatre cancellation figures", "Theatre Utilisation"),
         ("Could I get the medicines dispensed on each ward?", "Pharmacy Dispensing")]


def test_clm_picks_the_expected_dataset_from_the_catalogue(desk, openmetadata_url):
    registry = load_tools(desk)
    service_desk_tools.configure({"openmetadata_url": openmetadata_url}, desk.path.parent)
    entries = load_catalogue(desk, registry)
    question = {"type": "choice", "instructions": desk.catalogue.instructions,
                "criteria": {entry["name"]: entry["description"] for entry in entries}}
    router = LocalContrastiveRouter(desk.clm_model_id)
    picks = [router.ask(f"User: {text}", {"catalogue": question})["catalogue"] for text, _ in PICKS]
    for (text, want), got in zip(PICKS, picks):
        print(f"{got['choice']:<36}{got['confidence']:.2f}  (wanted {want}) {text}")
    # Measured 2 of 4 right, released and trained heads alike: CLM was not trained on these options.
    # What the design relies on is that a wrong pick is not confident enough to be used.
    threshold = desk.catalogue.confidence_threshold
    assert sum(got["choice"] == want for (_, want), got in zip(PICKS, picks)) >= 2
    assert all(got["choice"] == want for (_, want), got in zip(PICKS, picks) if got["confidence"] >= threshold)
