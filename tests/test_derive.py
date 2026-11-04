"""Tests for clm-derive: derivation with a canned extractor in place of a model, the layer it writes, and the
commands. No model, PostgreSQL or Ollama is needed."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument
import functools
import json
import sys
from datetime import date
from pathlib import Path

import httpx
import pytest
from fake_documents import FakeEmbedder, FakeStore

from clm_decision import derive, derived
from clm_decision.dmn import load_table
from clm_decision.environment import load_environment

HANDBOOK = Path(__file__).parent.parent / "examples" / "handbook" / "handbook.jsonl"
ON = date(2026, 12, 1)
LEAVE_2 = ("Up to 7 unused days carry into the next year; with 5 or more full years of service, up to 10. Days "
           "above the limit are forfeited on 31 December")
TOML = '''
[environment]
name = "Test"
description = "A test environment."
functions = "fx_functions"

[graph]
nodes = "nodes.csv"
edges = "edges.csv"

[documents]

[collections.handbook]
description = "The handbook."
embedding_model = "m"

[derived]
collection = "handbook"

[derived.decisions.carryover_limit]
description = "How many unused days of paid time off carry into next year, from full years of service."
inputs = { years_of_service = "number" }
outputs = { carryover_limit = "number" }
owner = "People Operations"
governs = ["benefits"]

[derived.facts.k401_2026.match_cap]
description = "The plan's cap on the company match, in dollars a plan year."
type = "number"
'''
CARRYOVER = {"statement": "Up to 7 unused days carry over, or 10 with 5 or more full years of service.",
             "reason": "The paid time off policy.",
             "rows": [{"inputs": {"years_of_service": ">= 5"}, "outputs": {"carryover_limit": 10},
                       "reason": "With 5 or more full years of service, up to 10 days."},
                      {"inputs": {"years_of_service": "-"}, "outputs": {"carryover_limit": 7},
                       "reason": "Up to 7 unused days carry over; days above the limit are forfeited on 31 December."}],
             "source": {"document": "hb-leave", "passages": ["(2)"], "quote": LEAVE_2}, "missing": []}
CAP = {"value": 6300, "source": {"document": "hb-401k-2026", "passages": ["(4)"],
                                 "quote": "The company match is capped at $6,300 a plan year"}, "missing": []}


class Canned:  # pylint: disable=too-few-public-methods
    """Answers by slot, in turn; records every prompt."""
    model = "canned"

    def __init__(self, **answers):
        self.answers = {name.replace("__", "."): list(given) for name, given in answers.items()}
        self.prompts: list[tuple[str, str]] = []

    def extract(self, system: str, prompt: str, schema: dict) -> dict:
        assert schema["type"] == "object"
        name = next(name for name in self.answers if f"Slot: {name}\n" in prompt)
        self.prompts.append((name, prompt))
        given = self.answers[name]
        return given.pop(0) if len(given) > 1 else given[0]


def store() -> FakeStore:
    found = FakeStore()
    found.replace("handbook", [json.loads(line) for line in HANDBOOK.read_text(encoding="utf-8").splitlines()], [],
                  {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "m", "dimensions": 26})
    return found


@pytest.fixture
def path(tmp_path, monkeypatch):
    (tmp_path / "fx_functions.py").write_text("", encoding="utf-8")
    (tmp_path / "nodes.csv").write_text("id,kind,name,match_cap\nk401_2026,plan,Plan,\n", encoding="utf-8")
    (tmp_path / "edges.csv").write_text("source,relation,target\n", encoding="utf-8")
    (tmp_path / "environment.toml").write_text(TOML, encoding="utf-8")
    monkeypatch.setattr(derive, "load_environment",
                        functools.partial(load_environment, store=store(), embedder=FakeEmbedder()))
    return tmp_path / "environment.toml"


def run(path, extractor, **options):
    return derive.run(path, "handbook", extractor, ON, **options)


def good():
    return Canned(carryover_limit=[CARRYOVER], k401_2026__match_cap=[CAP])


def test_run_fills_every_slot_and_checks_it(path):
    run(path, good())
    layer = derived.read(path.parent)
    assert layer["model"] == "canned" and layer["derived_at"] and layer["collection"] == "handbook"
    entry = layer["decisions"]["carryover_limit"]
    assert entry["status"] == "checked" and entry["check"]["passed"] and entry["attempts"] == 1
    assert entry["check"]["by"] == "clm-derive run"
    assert entry["source"]["edition"] == "2026-01-01" and entry["rule"]["cites"] == ["hb-leave"]
    assert entry["description"].startswith("How many unused days")
    assert (path.parent / "derived" / entry["dmn"]).is_file()
    assert layer["facts"]["k401_2026.match_cap"]["status"] == "checked"
    assert layer["facts"]["k401_2026.match_cap"]["value"] == 6300


def test_the_model_sees_only_documents_in_force(path):
    extractor = good()
    run(path, extractor)
    prompt = next(prompt for name, prompt in extractor.prompts if name == "k401_2026.match_cap")
    assert "hb-401k-2026" in prompt and "hb-401k-2024" not in prompt and "23,000" not in prompt


def test_a_failed_check_is_retried_once_with_its_failures(path):
    wrong = {**CAP, "value": 6400}
    extractor = Canned(carryover_limit=[CARRYOVER], k401_2026__match_cap=[wrong, CAP])
    run(path, extractor)
    entry = derived.read(path.parent)["facts"]["k401_2026.match_cap"]
    assert entry["status"] == "checked" and entry["attempts"] == 2
    [first, second] = [prompt for name, prompt in extractor.prompts if name == "k401_2026.match_cap"]
    assert "6400" not in first and "figure 6400" in second
    extractor = Canned(carryover_limit=[CARRYOVER], k401_2026__match_cap=[wrong])
    run(path, extractor, fresh=True)
    entry = derived.read(path.parent)["facts"]["k401_2026.match_cap"]
    assert entry["status"] == "proposed" and entry["attempts"] == 2 and not entry["check"]["passed"]
    assert len([name for name, _ in extractor.prompts if name == "k401_2026.match_cap"]) == 2


def approve_all(folder):
    for kind in ("decisions", "facts"):
        for name in derived.read(folder)[kind]:
            derived.set_status(folder, kind, name, "approved", by="kevin")


def test_run_keeps_an_approved_entry_on_an_unchanged_edition(path):
    run(path, good())
    derived.set_status(path.parent, "decisions", "carryover_limit", "approved", by="kevin")
    extractor = good()
    run(path, extractor)
    assert [name for name, _ in extractor.prompts] == ["k401_2026.match_cap"]
    assert derived.read(path.parent)["decisions"]["carryover_limit"]["decided_by"] == "kevin"


def test_run_derives_nothing_when_every_slot_is_approved(path):
    run(path, good())
    approve_all(path.parent)
    extractor = good()
    extractor.model = "another"
    run(path, extractor)
    assert not extractor.prompts and derived.read(path.parent)["model"] == "another"


def test_fresh_run_rederives_every_slot(path):
    run(path, good())
    approve_all(path.parent)
    extractor = good()
    run(path, extractor, fresh=True)
    assert sorted(name for name, _ in extractor.prompts) == ["carryover_limit", "k401_2026.match_cap"]
    assert derived.read(path.parent)["decisions"]["carryover_limit"]["status"] == "checked"


def test_run_rederives_an_entry_whose_edition_changed(path):
    run(path, good())
    approve_all(path.parent)
    layer = derived.read(path.parent)
    layer["decisions"]["carryover_limit"]["source"]["edition"] = "2025-01-01"
    derived.write(path.parent, layer)
    extractor = good()
    run(path, extractor)
    assert [name for name, _ in extractor.prompts] == ["carryover_limit"]
    assert derived.read(path.parent)["decisions"]["carryover_limit"]["status"] == "checked"


def test_run_writes_an_open_conflict_when_a_supplied_value_differs(path):
    (path.parent / "nodes.csv").write_text("id,kind,name,match_cap\nk401_2026,plan,Plan,5000\n", encoding="utf-8")
    run(path, good())
    conflict = derived.read(path.parent)["conflicts"]["k401_2026.match_cap"]
    assert (conflict["supplied"], conflict["derived"], conflict["status"]) == (5000, 6300, "open")


def test_rendered_dmn_loads_and_cites_its_document(path, tmp_path):
    env = derive.load_environment(path, today=ON)
    text = derive.render_dmn(env.slots["carryover_limit"], CARRYOVER["rows"], "hb-leave")
    (tmp_path / "t.dmn").write_text(text, encoding="utf-8")
    table = load_table(tmp_path / "t.dmn")
    assert table.inputs == ["years_of_service"] and table.outputs == ["carryover_limit", "reason", "citation"]
    assert [row.values for row in table.rules] == [
        (10, "With 5 or more full years of service, up to 10 days.", "hb-leave"),
        (7, "Up to 7 unused days carry over; days above the limit are forfeited on 31 December.", "hb-leave")]


def test_status_command_refuses_approving_a_failed_entry(path, capsys):
    run(path, Canned(carryover_limit=[CARRYOVER], k401_2026__match_cap=[{**CAP, "value": 6400}]))
    code = derive.main(["status", str(path), "facts", "k401_2026.match_cap", "approved", "--by", "kevin"])
    error = capsys.readouterr().err
    assert code == 2 and error.startswith("error: ") and error.count("\n") == 1
    assert derive.main(["status", str(path), "decisions", "carryover_limit", "approved", "--by", "kevin"]) == 0
    assert derived.read(path.parent)["decisions"]["carryover_limit"]["status"] == "approved"


def test_check_command_moves_a_corrected_entry_to_checked(path):
    run(path, Canned(carryover_limit=[CARRYOVER], k401_2026__match_cap=[{**CAP, "value": 6400}]))
    layer = derived.read(path.parent)
    # A correction, as the portal makes it: the value changed, status proposed and check null in one write
    layer["facts"]["k401_2026.match_cap"].update(value=6300, status="proposed", check=None)
    derived.write(path.parent, layer)
    assert derive.main(["check", str(path), "--as-of", "2026-12-01"]) == 0
    entry = derived.read(path.parent)["facts"]["k401_2026.match_cap"]
    assert entry["status"] == "checked" and entry["check"]["by"] == "clm-derive check"


class FakeHttp:  # pylint: disable=too-few-public-methods
    """Ollama's /api/chat, answering with the given content."""

    def __init__(self, content: str):
        self.content, self.bodies = content, []

    def post(self, url, json):  # pylint: disable=redefined-outer-name
        self.bodies.append(json)
        return httpx.Response(200, json={"message": {"content": self.content}, "done_reason": "stop"},
                              request=httpx.Request("POST", url))


def test_ollama_is_asked_without_thinking_and_with_a_token_cap():
    http = FakeHttp('{"value": 1}')
    assert derive.OllamaExtractor(http=http).extract("system", "prompt", {"type": "object"}) == {"value": 1}
    body = http.bodies[0]
    assert body["think"] is False and body["options"]["num_predict"] > 0 and body["format"] == {"type": "object"}


def test_an_answer_that_is_not_json_fails_the_check_without_stopping_the_run(path):
    class Broken(Canned):  # pylint: disable=missing-class-docstring
        def extract(self, system, prompt, schema):
            if "Slot: k401_2026.match_cap\n" in prompt:
                return derive.OllamaExtractor(http=FakeHttp('{"value": 63')).extract(system, prompt, schema)
            return super().extract(system, prompt, schema)
    run(path, Broken(carryover_limit=[CARRYOVER]))
    layer = derived.read(path.parent)
    assert layer["decisions"]["carryover_limit"]["status"] == "checked"
    entry = layer["facts"]["k401_2026.match_cap"]
    assert entry["status"] == "proposed" and "JSON" in entry["check"]["failures"][0] and entry["attempts"] == 2


def test_claude_backend_without_the_agent_extra_says_so_in_one_line(path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert derive.main(["run", str(path), "handbook", "--backend", "claude"]) == 2
    error = capsys.readouterr().err
    assert "agent extra" in error and error.count("\n") == 1
