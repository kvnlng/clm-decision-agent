"""Tests for the mechanical check of derived rules and facts, against the handbook example's documents, with a
stand-in store: every figure must be in the quote, the quote in the cited passages, and the edition the one in
force."""
# Test names say what they check.
# pylint: disable=missing-function-docstring
import json
from datetime import date
from pathlib import Path

import pytest
from fake_documents import FakeEmbedder, FakeStore
from test_derived_load import HEAD

from clm_decision import derive_check, derived
from clm_decision.environment import load_environment

HANDBOOK = Path(__file__).parent.parent / "examples" / "handbook" / "handbook.jsonl"
ON = date(2026, 12, 1)
LEAVE_2 = ("Up to 7 unused days carry into the next year; with 5 or more full years of service, up to 10. Days "
           "above the limit are forfeited on 31 December")
TOML = HEAD.replace("[collections.hb]", "[collections.handbook]") + '''
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

[derived.facts.k401_2026.first_tier_rate]
description = "The percentage of the first tier of pay contributed that the company matches."
type = "number"

[derived.facts.k401_2026.year]
description = "The plan year."
type = "number"
'''


def store() -> FakeStore:
    found = FakeStore()
    found.replace("handbook", [json.loads(line) for line in HANDBOOK.read_text(encoding="utf-8").splitlines()], [],
                  {"built_at": "2026-11-03T12:00:00Z", "source": "test", "model": "m", "dimensions": 26})
    return found


@pytest.fixture(name="env")
def environment(tmp_path):
    (tmp_path / "fx_functions.py").write_text("", encoding="utf-8")
    (tmp_path / "nodes.csv").write_text("id,kind,name,match_cap,year\nk401_2026,plan,Plan,,2026\n", encoding="utf-8")
    (tmp_path / "edges.csv").write_text("source,relation,target\n", encoding="utf-8")
    (tmp_path / "environment.toml").write_text(TOML, encoding="utf-8")
    return load_environment(tmp_path / "environment.toml", store=store(), embedder=FakeEmbedder(), today=ON)


def dmn(rows, input_name="years_of_service", citation="hb-leave") -> str:
    def row(index, test, limit, reason):
        cells = [test, str(limit), f'"{reason}"', f'"{citation}"']
        return (f'<rule id="r{index}"><inputEntry><text>{cells[0]}</text></inputEntry>'
                + "".join(f"<outputEntry><text>{cell}</text></outputEntry>" for cell in cells[1:]) + "</rule>")
    body = "".join(row(index, *cells) for index, cells in enumerate(rows))
    return ('<?xml version="1.0" encoding="UTF-8"?><definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" '
            'id="d" name="carryover_limit"><decision id="carryover_limit" name="carryover_limit">'
            '<decisionTable id="t" hitPolicy="FIRST">'
            f'<input id="i"><inputExpression typeRef="number"><text>{input_name}</text></inputExpression></input>'
            '<output id="o1" name="carryover_limit" typeRef="number"/><output id="o2" name="reason" typeRef="string"/>'
            f'<output id="o3" name="citation" typeRef="string"/>{body}</decisionTable></decision></definitions>')


GOOD = [("&gt;= 5", 10, "With 5 or more full years of service, up to 10 days."),
        ("-", 7, "Up to 7 unused days carry over; days above the limit are forfeited on 31 December.")]


def decision(env, table=dmn(GOOD), quote=LEAVE_2, passages=("(2)",), document="hb-leave", edition="2026-01-01",
             statement="Up to 7 unused days carry over, or 10 with 5 or more full years of service.", **extra):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rules = env.path.parent / "derived" / "rules"
    rules.mkdir(parents=True, exist_ok=True)
    (rules / "carryover_limit.dmn").write_text(table, encoding="utf-8")
    return {"description": "x", "dmn": "rules/carryover_limit.dmn", "collection": "handbook",
            "rule": {"statement": statement, "reason": "The paid time off policy.", "owner": "People Operations",
                     "version": edition, "governs": ["benefits"], "cites": [document]},
            "source": {"document": document, "passages": list(passages), "quote": quote, "edition": edition},
            "status": "proposed", "check": None, "decided_by": None, "decided_at": None, "note": None,
            "missing": [], **extra}


def fact(value=6300, quote="The company match is capped at $6,300 a plan year", passages=("(4)",),
         document="hb-401k-2026", edition="2026-01-01", node="k401_2026", prop="match_cap") -> dict:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return {"node": node, "property": prop, "value": value,
            "source": {"document": document, "passages": list(passages), "quote": quote, "edition": edition},
            "status": "proposed", "check": None, "decided_by": None, "decided_at": None, "note": None, "missing": []}


def failures(env, kind, name, entry):
    return derive_check.check(env, kind, name, entry, on=ON)


def test_numbers_reads_small_figures_ordinals_and_words():
    assert derive_check.numbers("Up to 7 days from the 90th day, after one full year, 60% of $24,500 or 4.5%; "
                                "twenty weeks") == [7, 90, 1, 60, 24500, 4.5, 20]


def test_the_hand_written_carryover_table_passes(env):
    assert failures(env, "decisions", "carryover_limit", decision(env)) == []


def test_numbers_reads_both_bounds_of_a_range():
    assert derive_check.numbers("[1..12]") == [1, 12]
    assert derive_check.numbers("(5..99]") == [5, 99]


def test_a_ranges_upper_bound_not_in_the_quote_fails(env):
    table = dmn([("[5..99]", 10, "With 5 or more full years of service, up to 10 days."), GOOD[1]])
    found = failures(env, "decisions", "carryover_limit", decision(env, table=table))
    assert any("figure 99" in failure for failure in found)


def test_a_quote_cut_mid_number_or_mid_word_fails(env):
    found = failures(env, "decisions", "carryover_limit", decision(env, quote=LEAVE_2 + " ... up to 1"))
    assert any("'up to 1'" in failure for failure in found)
    found = failures(env, "decisions", "carryover_limit", decision(env, quote=LEAVE_2 + " ... full yea"))
    assert any("'full yea'" in failure for failure in found)


def test_a_figure_in_the_rules_reason_must_be_in_the_quote(env):
    entry = decision(env)
    entry["rule"]["reason"] = "The paid time off policy, which allows 12 days."
    assert any("figure 12 (reason)" in failure for failure in failures(env, "decisions", "carryover_limit", entry))


def reason_failures(env, reason):
    entry = decision(env)
    entry["rule"]["reason"] = reason
    return failures(env, "decisions", "carryover_limit", entry)


def test_an_iso_date_in_a_reason_is_not_a_figure(env):
    assert reason_failures(env, "The leave policy, in force on 2026-12-01.") == []


def test_a_section_reference_in_a_reason_is_not_a_figure(env):
    assert reason_failures(env, "Handbook § 5 (2), as § 5(2) and Section 5(2) say, and § 4.") == []


def test_the_plans_name_401k_is_not_a_figure(env):
    entry = decision(env, statement="Under the 401(k) plan, up to 7 unused days carry over, or 10 with 5 or more.")
    assert failures(env, "decisions", "carryover_limit", entry) == []


def test_a_real_figure_beside_an_exempt_one_still_fails(env):
    found = reason_failures(env, "§ 5(2), in force on 2026-12-01, under the 401(k) plan: 12 days.")
    assert [failure for failure in found if failure.startswith("figure")] == ["figure 12 (reason) is not in the quote"]


def test_a_quote_not_in_the_passage_fails(env):
    found = failures(env, "decisions", "carryover_limit", decision(env, quote="Up to 8 unused days carry over"))
    assert any("quote" in failure for failure in found)


def test_a_quote_in_two_pieces_across_cited_passages_passes(env):
    quote = "You earn 15 days of paid time off a year ... " + LEAVE_2
    assert failures(env, "decisions", "carryover_limit", decision(env, quote=quote, passages=("(1)", "(2)"))) == []


def test_an_uncited_passage_fails(env):
    assert any("quote" in failure for failure in failures(env, "decisions", "carryover_limit",
                                                         decision(env, passages=("(1)",))))
    assert any("(9)" in failure for failure in failures(env, "decisions", "carryover_limit",
                                                       decision(env, passages=("(2)", "(9)"))))


def test_a_figure_not_in_the_quote_fails(env):
    table = dmn([("&gt;= 5", 10, "Up to 10 days."), ("-", 8, "Up to 8 days.")])
    found = failures(env, "decisions", "carryover_limit", decision(env, table=table))
    assert any("8" in failure and "quote" in failure for failure in found)
    found = failures(env, "decisions", "carryover_limit", decision(env, statement="Up to 12 days carry over."))
    assert any("12" in failure for failure in found)


def test_zero_is_exempt(env):
    table = dmn([("&lt; 0", 0, "None."), *GOOD])
    assert failures(env, "decisions", "carryover_limit", decision(env, table=table)) == []


def test_a_percentage_matches_its_figure(env):
    entry = fact(100, "Meridian matches 100% of the first 3% of pay you contribute", ("(3)",), prop="first_tier_rate")
    assert failures(env, "facts", "k401_2026.first_tier_rate", entry) == []


def test_the_superseded_401k_edition_fails(env):
    entry = fact(23000, "which for 2024 is $23,000", ("(2)",), document="hb-401k-2024", edition="2024-01-01")
    assert any("not in force" in failure for failure in failures(env, "facts", "k401_2026.match_cap", entry))


def test_a_wrong_edition_date_fails(env):
    found = failures(env, "facts", "k401_2026.match_cap", fact(edition="2025-01-01"))
    assert any("edition" in failure for failure in found)


def test_an_unknown_node_or_property_fails(env):
    assert any("k401_2099" in failure for failure in failures(env, "facts", "k401_2026.match_cap",
                                                             fact(node="k401_2099")))
    assert any("irs_limit" in failure for failure in failures(env, "facts", "k401_2026.match_cap",
                                                             fact(prop="irs_limit")))
    assert any("type" in failure for failure in failures(env, "facts", "k401_2026.match_cap", fact(value="6300")))


def test_dmn_inputs_must_be_the_slots(env):
    found = failures(env, "decisions", "carryover_limit", decision(env, table=dmn(GOOD, input_name="years")))
    assert any("inputs" in failure for failure in found)


def test_an_unparsable_dmn_fails(env):
    assert any("dmn" in failure.lower() for failure in failures(env, "decisions", "carryover_limit",
                                                               decision(env, table="<definitions>")))


def test_citation_cells_name_only_the_source(env):
    found = failures(env, "decisions", "carryover_limit", decision(env, table=dmn(GOOD, citation="hb-dress")))
    assert any("hb-dress" in failure for failure in found)


def test_missing_ontology_fails(env):
    found = failures(env, "decisions", "carryover_limit", decision(env, missing=["region"]))
    assert any("region" in failure for failure in found)


def layer_with(env, **entries) -> Path:
    folder = env.path.parent
    layer = derived.empty("handbook")
    for name, (kind, entry) in entries.items():
        layer[kind][name.replace("__", ".")] = entry
    derived.write(folder, layer)
    return folder


def test_recheck_never_touches_approved_or_rejected(env):
    bad = fact(edition="2025-01-01")
    folder = layer_with(env, carryover_limit=("decisions", decision(env, status="approved")),
                        k401_2026__match_cap=("facts", {**bad, "status": "rejected"}),
                        k401_2026__first_tier_rate=("facts", fact(100, "Meridian matches 100% of the first 3%",
                                                                  ("(3)",), prop="first_tier_rate")))
    counts = derive_check.recheck(folder, env, on=ON)
    layer = derived.read(folder)
    assert layer["decisions"]["carryover_limit"]["status"] == "approved"
    assert layer["decisions"]["carryover_limit"]["check"] is None
    assert layer["facts"]["k401_2026.match_cap"]["status"] == "rejected"
    assert layer["facts"]["k401_2026.first_tier_rate"]["status"] == "checked"
    assert counts == {"checked": 1, "proposed": 0, "stale": 0, "untouched": 2}


def test_recheck_sends_an_approved_entry_on_a_replaced_edition_back_to_proposed(env):
    stale = {**decision(env, status="approved", edition="2025-01-01"), "decided_by": "kevin"}
    folder = layer_with(env, carryover_limit=("decisions", stale))
    counts = derive_check.recheck(folder, env, on=ON)
    entry = derived.read(folder)["decisions"]["carryover_limit"]
    assert entry["status"] == "proposed" and entry["decided_by"] == "kevin"
    assert "newer edition" in entry["check"]["failures"][0] and counts["stale"] == 1


def test_recheck_accepts_a_null_check_after_a_correction(env):
    folder = layer_with(env, carryover_limit=("decisions", decision(env)))
    assert derived.read(folder)["decisions"]["carryover_limit"]["check"] is None
    derive_check.recheck(folder, env, on=ON)
    entry = derived.read(folder)["decisions"]["carryover_limit"]
    assert entry["status"] == "checked" and entry["check"]["passed"] and entry["check"]["by"] == "clm-derive check"


def test_a_differing_supplied_value_is_an_open_conflict(env):
    layer = derived.empty("handbook")
    layer["facts"]["k401_2026.year"] = fact(2024, "which for 2024 is", ("(2)",), prop="year")
    layer["facts"]["k401_2026.match_cap"] = fact()
    found = derive_check.conflicts(env, layer)
    assert list(found) == ["k401_2026.year"]
    assert found["k401_2026.year"] | {"source": None} == {
        "target": "facts", "name": "k401_2026.year", "supplied": 2026, "derived": 2024, "source": None,
        "status": "open", "decided_by": None, "decided_at": None, "note": None}
