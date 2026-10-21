"""End-to-end tests with the real CLM, through the MCP server. Run with: uv run pytest -m slow"""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import shutil
from pathlib import Path

import pytest
import service_desk_tools
from desk_routing_cases import CASES
from mcp import Client

import server

pytestmark = pytest.mark.slow

# Untrained, CLM's routing was weak (5 of 20 when this floor was set); this guards against the
# scenario's wording collapsing to one route. With the example's trained heads it does much better.
ROUTING_FLOOR = 5


@pytest.fixture(scope="module")
def desk(tmp_path_factory):
    """The example loaded from a copy, so a test cannot change its files; -> (scenario, router, tools)."""
    example = Path(__file__).parent.parent / "examples" / "service_desk"
    folder = tmp_path_factory.mktemp("desk")
    for name in ("scenario.toml", "queues.json"):
        shutil.copy(example / name, folder / name)
    if (example / "scenario.heads").is_dir():
        shutil.copytree(example / "scenario.heads", folder / "scenario.heads")
    return server.load(str(folder / "scenario.toml"))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_real_server_decides(desk, tracker_server, tracker_url):
    scenario, router, tools = desk
    service_desk_tools.SETTINGS["tracker_url"] = tracker_url
    before = set(tracker_server.tracker.issues)
    async with Client(server.build_server(scenario, router, tools, review=False)) as client:
        results = [((await client.call_tool("decide", {"text": text})).structured_content, route)
                   for text, route in CASES]
    correct = sum(got["route"] == want for got, want in results)
    print(f"routing through the server: {correct} of {len(CASES)} correct")
    assert correct >= ROUTING_FLOOR

    oncology = results[1][0]            # "Can I get into the oncology registry?"
    assert oncology["informed"][0]["facts"][0] == "Data Governance handles this kind of request."
    tickets = [action for got, _ in results for action in got["actions"] if action["status"] == "done"]
    assert len(set(tracker_server.tracker.issues) - before) == len(tickets) > 0
    out_of_scope = [got for got, want in results if want == "Out_Of_Scope" and got["route"] == "Out_Of_Scope"]
    assert all(not got["actions"] for got in out_of_scope)
