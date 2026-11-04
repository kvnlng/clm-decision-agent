"""Tests for clm-derive: derivation with a canned extractor in place of a model, the layer it writes, and the
commands. No model, PostgreSQL or Ollama is needed."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument
import dataclasses
import functools
import sys
import types

import httpx
import pytest
import test_derive_check
from fake_documents import FakeEmbedder
from test_derive_check import LEAVE_2, ON, store

from clm_decision import derive, derived
from clm_decision.dmn import load_table
from clm_decision.environment import load_environment

# The check tests' environment with its first two slots only: the carryover table and the match cap
TOML = test_derive_check.TOML.split("\n[derived.facts.k401_2026.first_tier_rate]", maxsplit=1)[0]
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
    assert entry["check"]["dmn_sha256"] and entry["check"]["dmn_sha256"] == derived.dmn_digest(path.parent, entry)
    assert layer["facts"]["k401_2026.match_cap"]["status"] == "checked"
    assert layer["facts"]["k401_2026.match_cap"]["value"] == 6300


def test_the_model_sees_only_documents_in_force(path):
    extractor = good()
    run(path, extractor)
    prompt = next(prompt for name, prompt in extractor.prompts if name == "k401_2026.match_cap")
    assert "hb-401k-2026" in prompt and "hb-401k-2024" not in prompt and "23,000" not in prompt


def test_the_prompt_gives_each_documents_id_on_its_own_line(path):
    extractor = good()
    run(path, extractor)
    prompt = next(prompt for name, prompt in extractor.prompts if name == "k401_2026.match_cap")
    assert "\nDocument id: hb-401k-2026\n" in prompt and "the document's id" in derive.SYSTEM


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


def test_the_command_says_which_slots_it_kept_and_when_it_derived_nothing(path, monkeypatch, capsys):
    run(path, good())
    approve_all(path.parent)
    monkeypatch.setattr(derive, "_extractor", lambda args: good())
    assert derive.main(["run", str(path), "handbook", "--as-of", "2026-12-01"]) == 0
    out = capsys.readouterr().out
    assert "carryover_limit: approved, kept" in out
    assert "Nothing derived: every slot is approved or rejected on an unchanged edition" in out


def test_fresh_run_rederives_every_slot(path):
    run(path, good())
    approve_all(path.parent)
    extractor = good()
    run(path, extractor, fresh=True)
    assert sorted(name for name, _ in extractor.prompts) == ["carryover_limit", "k401_2026.match_cap"]
    assert derived.read(path.parent)["decisions"]["carryover_limit"]["status"] == "checked"


def test_a_run_that_stops_part_way_leaves_the_previous_tables_and_layer_as_they_were(path):
    run(path, good())
    rules = path.parent / "derived" / "rules"
    table, layer = (rules / "carryover_limit.dmn").read_bytes(), derived.read(path.parent)

    class Stops(Canned):  # pylint: disable=missing-class-docstring
        def extract(self, system, prompt, schema):
            if "Slot: k401_2026.match_cap\n" in prompt:
                raise derive.DeriveError("cannot reach Ollama")
            return super().extract(system, prompt, schema)
    with pytest.raises(derive.DeriveError):
        run(path, Stops(carryover_limit=[SHIFTED]), fresh=True)
    assert (rules / "carryover_limit.dmn").read_bytes() == table and derived.read(path.parent) == layer
    assert sorted(item.name for item in (path.parent / "derived").iterdir()) == ["derived.json", "rules"]


def test_a_finished_run_replaces_the_tables_and_leaves_nothing_aside(path):
    run(path, good())
    run(path, Canned(carryover_limit=[SHIFTED], k401_2026__match_cap=[CAP]), fresh=True)
    entry = derived.read(path.parent)["decisions"]["carryover_limit"]
    assert entry["dmn"] == "rules/carryover_limit.dmn"
    assert ">= 4" in (path.parent / "derived" / entry["dmn"]).read_text(encoding="utf-8").replace("&gt;", ">")
    assert sorted(item.name for item in (path.parent / "derived").iterdir()) == ["derived.json", "rules"]


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


class FakeAnthropic:  # pylint: disable=too-few-public-methods
    """The Anthropic client's messages.create, answering with the given text, or raising the given error."""

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply, self.error, self.messages = reply, error, self

    def create(self, **request):
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=self.reply)])


def test_a_claude_answer_that_is_not_json_fails_that_answer_without_stopping_the_run(path):
    class Broken(Canned):  # pylint: disable=missing-class-docstring
        def extract(self, system, prompt, schema):
            if "Slot: k401_2026.match_cap\n" in prompt:
                return derive.ClaudeExtractor(FakeAnthropic('{"value": 63'), "claude-x").extract(system, prompt,
                                                                                                   schema)
            return super().extract(system, prompt, schema)
    run(path, Broken(carryover_limit=[CARRYOVER]))
    layer = derived.read(path.parent)
    assert layer["decisions"]["carryover_limit"]["status"] == "checked"
    entry = layer["facts"]["k401_2026.match_cap"]
    assert entry["status"] == "proposed" and entry["attempts"] == 2
    assert entry["check"]["failures"][0].startswith("answer: Claude (claude-x) did not answer in JSON: ")


def test_claude_that_cannot_be_reached_still_stops_the_run():
    claude = derive.ClaudeExtractor(FakeAnthropic(error=RuntimeError("connection refused")), "claude-x")
    with pytest.raises(derive.DeriveError) as raised:
        claude.extract("system", "prompt", {"type": "object"})
    assert not isinstance(raised.value, derive.AnswerError) and "connection refused" in str(raised.value)


def test_claude_backend_without_the_agent_extra_says_so_in_one_line(path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert derive.main(["run", str(path), "handbook", "--backend", "claude"]) == 2
    error = capsys.readouterr().err
    assert "agent extra" in error and error.count("\n") == 1


SHIFTED = {**CARRYOVER, "rows": [{**CARRYOVER["rows"][0], "inputs": {"years_of_service": ">= 4"}},
                                 CARRYOVER["rows"][1]]}


def second(**answers):
    reader = Canned(**answers)
    reader.model = "second"
    return reader


@pytest.fixture
def read_again(path, monkeypatch):
    """The environment with a second reader configured, answered by the returned list's last Canned."""
    path.write_text(path.read_text(encoding="utf-8").replace(
        '[derived]\ncollection = "handbook"',
        '[derived]\ncollection = "handbook"\nsecond_reader = { backend = "ollama", model = "second" }'),
        encoding="utf-8")
    readers = [second(carryover_limit=[CARRYOVER], k401_2026__match_cap=[CAP])]
    monkeypatch.setattr(derive, "reader_for", lambda env: readers[-1] if env.second_reader else None)
    return readers


def test_run_with_a_second_reader_records_that_it_agrees(path, read_again):
    run(path, good())
    entry = derived.read(path.parent)["decisions"]["carryover_limit"]
    assert entry["status"] == "checked" and entry["check"]["passed"]
    assert entry["check"]["second_reading"] == {"model": "second", "agreed": True, "differences": []}


def test_a_second_reading_that_disagrees_leaves_the_entry_proposed_saying_where(path, read_again):
    read_again.append(second(carryover_limit=[SHIFTED], k401_2026__match_cap=[{**CAP, "value": 6000}]))
    first = good()
    run(path, first)
    layer = derived.read(path.parent)
    entry = layer["decisions"]["carryover_limit"]
    assert entry["status"] == "proposed" and entry["attempts"] == 1
    assert entry["check"]["failures"] == ["a second reading by second disagrees: at years_of_service=4, 7 against 10; "
                                          "at years_of_service=4.5, 7 against 10"]
    assert entry["check"]["second_reading"]["differences"] == [
        {"inputs": {"years_of_service": years}, "first": {"carryover_limit": 7}, "second": {"carryover_limit": 10}}
        for years in (4, 4.5)]
    assert layer["facts"]["k401_2026.match_cap"]["check"]["failures"] == [
        "a second reading by second disagrees: 6300 against 6000"]
    assert len(first.prompts) == 2


def test_the_second_reader_reads_the_cited_document_and_never_the_first_answer(path, read_again):
    run(path, good())
    [(_, prompt)] = [item for item in read_again[-1].prompts if item[0] == "carryover_limit"]
    assert prompt.count("Document id: ") == 1 and "\nDocument id: hb-leave\n" in prompt
    assert "Your answer was" not in prompt and "Slot: carryover_limit\n" in prompt
    # Exactly the prompt a first reading of that one document gets: its slot and the document whole, so neither the
    # first answer's passages and quote nor its statement, reasons or rows reach the second reader
    env = derive.load_environment(path, today=ON)
    document = env.store.document("handbook", "hb-leave", None)
    assert prompt == derive._prompt(env.slots["carryover_limit"], [document], ON)  # pylint: disable=protected-access
    assert CARRYOVER["statement"] not in prompt and CARRYOVER["rows"][0]["reason"] not in prompt


def test_a_second_reading_that_cannot_be_made_fails_the_check(path, read_again):
    class Down(Canned):  # pylint: disable=missing-class-docstring
        model = "second"

        def extract(self, system, prompt, schema):
            raise derive.DeriveError("cannot reach Ollama")
    read_again.append(Down())
    run(path, good())
    entry = derived.read(path.parent)["facts"]["k401_2026.match_cap"]
    assert entry["status"] == "proposed"
    assert entry["check"]["failures"] == ["a second reading by second could not be made: cannot reach Ollama"]
    assert entry["check"]["second_reading"]["error"] == "cannot reach Ollama"


def test_a_first_reading_with_no_answer_is_not_read_again_as_a_disagreement(path, read_again):
    class Broken(Canned):  # pylint: disable=missing-class-docstring
        def extract(self, system, prompt, schema):
            return derive.OllamaExtractor(http=FakeHttp('{"value": 63')).extract(system, prompt, schema)
    run(path, Broken(carryover_limit=[CARRYOVER]))
    layer = derived.read(path.parent)
    for kind, name in (("decisions", "carryover_limit"), ("facts", "k401_2026.match_cap")):
        failures = layer[kind][name]["check"]["failures"]
        assert layer[kind][name]["status"] == "proposed" and "JSON" in failures[0]
        assert failures[-1] == ("a second reading by second could not be made: the first reading gave nothing to "
                                "compare")


def test_a_second_reader_that_is_the_layers_model_is_refused(path, read_again, monkeypatch, capsys):
    read_again[-1].model = "canned"
    monkeypatch.setattr(derive, "_extractor", lambda args: good())
    assert derive.main(["run", str(path), "handbook", "--as-of", "2026-12-01"]) == 2
    assert "second reader" in capsys.readouterr().err and not derived.path(path.parent).exists()
    read_again[-1].model = "second"
    run(path, good())
    read_again[-1].model = "canned"
    assert derive.main(["check", str(path), "--as-of", "2026-12-01"]) == 2
    error = capsys.readouterr().err
    assert "second reader canned" in error and error.count("\n") == 1


@pytest.mark.parametrize("first, again", [("gemma4:26b", "gemma4:26B"), ("gemma4:latest", "gemma4"),
                                          ("gemma4:26b", "gemma4"), ("gemma4", " gemma4:e4b "),
                                          ("claude-opus-5-5-20261001", "claude-opus-5-5"),
                                          ("Claude-Opus-5-5", "claude-opus-5-5-20261001")])
def test_another_tag_or_date_of_the_layers_model_is_the_same_reader(first, again):
    reader = second()
    reader.model = again
    with pytest.raises(derive.DeriveError, match=f"second reader {again.strip()} .* {first}"):
        derive.refuse_same_model(reader, first)


@pytest.mark.parametrize("first, again", [("gemma4:26b", "qwen3:14b"), ("gemma4:26b", "claude-opus-5-5"),
                                          ("claude-opus-5-5", "claude-sonnet-5"), ("granite4:micro", "gemma4")])
def test_a_different_model_family_may_read_again(first, again):
    reader = second()
    reader.model = again
    derive.refuse_same_model(reader, first)


def test_check_command_makes_the_second_reading_and_says_how_it_went(path, read_again, capsys):
    run(path, good())
    read_again.append(second(carryover_limit=[SHIFTED], k401_2026__match_cap=[CAP]))
    assert derive.main(["check", str(path), "--as-of", "2026-12-01"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["1 checked, 1 proposed, 0 stale, 0 untouched",
                   "second reading by second: 1 agreed, 1 disagreed, 0 not made"]
    assert derived.read(path.parent)["decisions"]["carryover_limit"]["status"] == "proposed"


def test_check_command_without_a_second_reader_says_there_is_none(path, capsys):
    run(path, good())
    assert derive.main(["check", str(path), "--as-of", "2026-12-01"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "no second reading: [derived] names no second_reader"
    assert "second_reading" not in derived.read(path.parent)["decisions"]["carryover_limit"]["check"]


def test_the_second_reader_is_the_backend_and_model_the_environment_names(path):
    env = derive.load_environment(path, today=ON)
    assert derive.reader_for(env) is None
    reader = derive.reader_for(dataclasses.replace(env, second_reader={"backend": "ollama",
                                                                       "model": "granite4:micro"}))
    assert isinstance(reader, derive.OllamaExtractor) and reader.model == "granite4:micro"
