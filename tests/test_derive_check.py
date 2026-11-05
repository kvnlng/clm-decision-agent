"""Tests for the mechanical check of derived rules and facts, against the handbook example's documents, with a
stand-in store: every figure must be in the quote, the quote in the cited passages, and the edition the one in
force."""
# Test names say what they check.
# pylint: disable=missing-function-docstring
import dataclasses
import json
from datetime import date
from pathlib import Path

import pytest
from fake_documents import FakeEmbedder, FakeStore
from test_derived_load import HEAD

from clm_decision import derive, derive_check, derived
from clm_decision.dmn import DmnError, load_table
from clm_decision.environment import Slot, load_environment

HANDBOOK = Path(__file__).parent.parent / "examples" / "handbook" / "handbook.jsonl"
ON = date(2026, 12, 1)
LEAVE_2 = ("Up to 7 unused days carry into the next year; with 5 or more full years of service, up to 10. Days "
           "above the limit are forfeited on 31 December")
TOML = HEAD.replace("[collections.hb]", "[collections.handbook]") + '''
[derived]
collection = "handbook"

[derived.decisions.carryover_limit]
description = "How many unused days of paid time off carry into next year, from full years of service."
inputs = { years_of_service = "integer" }
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


def test_a_bare_year_in_a_reason_is_not_a_figure(env):
    assert reason_failures(env, "The 2026 leave policy, which replaced the 1999 edition.") == []


def test_a_bare_section_reference_in_a_reason_is_not_a_figure(env):
    assert reason_failures(env, "The condition in 6(3) and in 5(2)(a) is not an input.") == []


def test_a_year_as_dollars_days_or_percent_is_still_a_figure(env):
    for text in ("A bonus of $2026.", "Up to 2026 days.", "Up to 2026%.", "Up to 2026.5 hours."):
        assert any("figure 2026" in failure for failure in reason_failures(env, text)), text


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
    approved = decision(env, status="approved")
    approved["check"] = {"passed": True, "failures": [], "dmn_sha256": derived.dmn_digest(env.path.parent, approved)}
    folder = layer_with(env, carryover_limit=("decisions", approved),
                        k401_2026__match_cap=("facts", {**bad, "status": "rejected"}),
                        k401_2026__first_tier_rate=("facts", fact(100, "Meridian matches 100% of the first 3%",
                                                                  ("(3)",), prop="first_tier_rate")))
    counts = derive_check.recheck(folder, env, on=ON)
    layer = derived.read(folder)
    assert layer["decisions"]["carryover_limit"]["status"] == "approved"
    assert layer["decisions"]["carryover_limit"]["check"] == approved["check"]
    assert layer["facts"]["k401_2026.match_cap"]["status"] == "rejected"
    assert layer["facts"]["k401_2026.first_tier_rate"]["status"] == "checked"
    assert counts == {"checked": 1, "proposed": 0, "stale": 0, "untouched": 2, "undigested": 0}


@pytest.mark.parametrize("old_check", [None, {"passed": True, "failures": []}])
def test_an_approved_table_checked_before_digests_were_kept_is_sent_back_once_and_checked(env, old_check):
    folder = layer_with(env, carryover_limit=("decisions", {**decision(env, status="approved"), "check": old_check,
                                                            "decided_by": "kevin"}))
    counts = derive_check.recheck(folder, env, on=ON)
    entry = derived.read(folder)["decisions"]["carryover_limit"]
    assert entry["status"] == "checked" and entry["decided_by"] == "kevin" and "no digest" in entry["note"]
    assert entry["check"]["dmn_sha256"] == derived.dmn_digest(folder, entry)
    assert counts["undigested"] == 1 and counts["checked"] == 1 and counts["stale"] == 0
    derived.set_status(folder, "decisions", "carryover_limit", "approved", by="kevin")
    assert derive_check.recheck(folder, env, on=ON)["untouched"] == 1


def test_recheck_sends_an_approved_entry_on_a_replaced_edition_back_to_proposed(env):
    stale = {**decision(env, status="approved", edition="2025-01-01"), "decided_by": "kevin"}
    folder = layer_with(env, carryover_limit=("decisions", stale))
    counts = derive_check.recheck(folder, env, on=ON)
    entry = derived.read(folder)["decisions"]["carryover_limit"]
    assert entry["status"] == "proposed" and entry["decided_by"] == "kevin"
    assert "newer edition" in entry["check"]["failures"][0] and counts["stale"] == 1


def test_recheck_sends_an_approved_entry_whose_table_changed_back_and_checks_it_again(env):
    approved = {**decision(env, status="approved"), "decided_by": "kevin"}
    approved["check"] = {"passed": True, "failures": [], "dmn_sha256": "0" * 64}
    folder = layer_with(env, carryover_limit=("decisions", approved))
    counts = derive_check.recheck(folder, env, on=ON)
    entry = derived.read(folder)["decisions"]["carryover_limit"]
    assert entry["status"] == "checked" and entry["decided_by"] == "kevin"
    assert "changed since its check" in entry["note"]
    assert entry["check"]["dmn_sha256"] == derived.dmn_digest(folder, entry)
    assert counts["stale"] == 1 and counts["checked"] == 1


def test_recheck_accepts_a_null_check_after_a_correction(env):
    folder = layer_with(env, carryover_limit=("decisions", decision(env)))
    assert derived.read(folder)["decisions"]["carryover_limit"]["check"] is None
    derive_check.recheck(folder, env, on=ON)
    entry = derived.read(folder)["decisions"]["carryover_limit"]
    assert entry["status"] == "checked" and entry["check"]["passed"] and entry["check"]["by"] == "clm-derive check"


def reading(agreed, model="m"):
    def second(_kind, _name, _entry):
        record = {"model": model, "agreed": agreed, "differences": [] if agreed else [{"first": 6300, "second": 63}]}
        return record, [] if agreed else [derive_check.disagreement(model, record["differences"])]
    return second


def test_a_disagreeing_second_reading_fails_the_check_and_is_recorded(env):
    folder = layer_with(env, k401_2026__match_cap=("facts", fact()))
    counts = derive_check.recheck(folder, env, on=ON, second=reading(False))
    entry = derived.read(folder)["facts"]["k401_2026.match_cap"]
    assert entry["status"] == "proposed" and not entry["check"]["passed"]
    assert entry["check"]["failures"] == ["a second reading by m disagrees: 6300 against 63"]
    assert entry["check"]["second_reading"]["agreed"] is False
    assert counts == {"checked": 0, "proposed": 1, "stale": 0, "untouched": 0, "undigested": 0, "agreed": 0,
                      "disagreed": 1,
                      "not_made": 0}


def test_an_agreeing_second_reading_lets_a_passing_entry_be_checked(env):
    folder = layer_with(env, k401_2026__match_cap=("facts", fact()))
    counts = derive_check.recheck(folder, env, on=ON, second=reading(True))
    entry = derived.read(folder)["facts"]["k401_2026.match_cap"]
    assert entry["status"] == "checked" and entry["check"]["second_reading"]["agreed"] is True
    assert counts["agreed"] == 1


def test_a_recorded_disagreement_keeps_failing_when_the_second_reader_is_removed(env):
    folder = layer_with(env, k401_2026__match_cap=("facts", fact()))
    derive_check.recheck(folder, env, on=ON, second=reading(False))
    derive_check.recheck(folder, env, on=ON)
    entry = derived.read(folder)["facts"]["k401_2026.match_cap"]
    assert entry["status"] == "proposed" and not entry["check"]["passed"]
    assert entry["check"]["failures"] == ["an earlier second reading by m disagrees: 6300 against 63"]
    assert entry["check"]["second_reading"]["agreed"] is False


def test_an_unusable_earlier_second_reading_keeps_failing_without_a_reader(env):
    def down(_kind, _name, _entry):
        return {"model": "m", "agreed": False, "differences": [], "error": "unreachable"}, ["x"]
    folder = layer_with(env, k401_2026__match_cap=("facts", fact()))
    derive_check.recheck(folder, env, on=ON, second=down)
    derive_check.recheck(folder, env, on=ON)
    entry = derived.read(folder)["facts"]["k401_2026.match_cap"]
    assert entry["check"]["failures"] == ["an earlier second reading by m could not be made: unreachable"]


def test_a_later_agreeing_reading_or_a_correction_clears_an_earlier_disagreement(env):
    folder = layer_with(env, k401_2026__match_cap=("facts", fact()))
    derive_check.recheck(folder, env, on=ON, second=reading(False))
    derive_check.recheck(folder, env, on=ON, second=reading(True))
    assert derived.read(folder)["facts"]["k401_2026.match_cap"]["status"] == "checked"
    derive_check.recheck(folder, env, on=ON, second=reading(False))
    layer = derived.read(folder)
    layer["facts"]["k401_2026.match_cap"].update(status="proposed", check=None)
    derived.write(folder, layer)
    derive_check.recheck(folder, env, on=ON)
    entry = derived.read(folder)["facts"]["k401_2026.match_cap"]
    assert entry["status"] == "checked" and "second_reading" not in entry["check"]


def test_an_earlier_agreeing_reading_is_not_carried_into_a_check_without_a_reader(env):
    # It compared the entry as it was then, so a later check must not vouch for it with that check's time
    folder = layer_with(env, k401_2026__match_cap=("facts", fact()))
    derive_check.recheck(folder, env, on=ON, second=reading(True))
    derive_check.recheck(folder, env, on=ON)
    entry = derived.read(folder)["facts"]["k401_2026.match_cap"]
    assert entry["status"] == "checked" and "second_reading" not in entry["check"]
    with pytest.raises(derived.DerivedError, match="no second reading by m"):
        derived.set_status(folder, "facts", "k401_2026.match_cap", "approved", by="kevin", second_reader="m")


def test_an_entry_whose_slot_is_gone_fails_without_a_second_reading(env):
    made = []

    def second(kind, name, entry):
        made.append(name)
        return reading(True)(kind, name, entry)
    gone = fact(prop="gone")
    folder = layer_with(env, k401_2026__gone=("facts", gone), k401_2026__match_cap=("facts", fact()))
    counts = derive_check.recheck(folder, env, on=ON, second=second)
    entry = derived.read(folder)["facts"]["k401_2026.gone"]
    assert entry["status"] == "proposed" and entry["check"]["failures"] == [
        "ontology: the environment has no facts slot k401_2026.gone"]
    assert made == ["k401_2026.match_cap"] and counts["not_made"] == 1 and counts["agreed"] == 1


def test_a_stale_entry_is_not_sent_back_again_while_it_is_still_stale(env):
    folder = layer_with(env, carryover_limit=("decisions", {**decision(env, status="approved", edition="2025-01-01"),
                                                            "decided_by": "kevin", "note": "matches"}))
    assert derive_check.recheck(folder, env, on=ON)["stale"] == 1
    once = derived.read(folder)["decisions"]["carryover_limit"]
    assert once["note"].count("back to proposed") == 1
    assert derive_check.recheck(folder, env, on=ON)["stale"] == 1
    assert derived.read(folder)["decisions"]["carryover_limit"] == once


def test_an_entry_whose_document_is_out_of_force_is_sent_back_once_over_several_days(env):
    # The reason names no date, so a check on a later day finds it still stale for the same reason
    entry = {**fact(23000, "which for 2024 is $23,000", ("(2)",), document="hb-401k-2024", edition="2024-01-01"),
             "status": "approved", "decided_by": "kevin", "note": "matches"}
    folder = layer_with(env, k401_2026__match_cap=("facts", entry))
    derive_check.recheck(folder, env, on=ON)
    once = derived.read(folder)["facts"]["k401_2026.match_cap"]
    assert once["note"].count("back to proposed") == 1 and "no longer in force" in once["note"]
    derive_check.recheck(folder, env, on=date(2026, 12, 2))
    assert derived.read(folder)["facts"]["k401_2026.match_cap"] == once


def test_a_second_reading_is_made_on_an_entry_the_mechanical_check_fails_too(env):
    folder = layer_with(env, k401_2026__match_cap=("facts", fact(quote="The match is capped at $6,300")))
    derive_check.recheck(folder, env, on=ON, second=reading(False))
    entry = derived.read(folder)["facts"]["k401_2026.match_cap"]
    assert len(entry["check"]["failures"]) == 2 and "second reading" in entry["check"]["failures"][-1]


def test_no_second_reading_is_made_on_approved_or_rejected_entries(env):
    made = []
    folder = layer_with(env, k401_2026__match_cap=("facts", {**fact(), "status": "approved"}))
    derive_check.recheck(folder, env, on=ON, second=lambda *args: made.append(args))
    assert not made


def test_a_differing_supplied_value_is_an_open_conflict(env):
    layer = derived.empty("handbook")
    layer["facts"]["k401_2026.year"] = fact(2024, "which for 2024 is", ("(2)",), prop="year")
    layer["facts"]["k401_2026.match_cap"] = fact()
    found = derive_check.conflicts(env, layer)
    assert list(found) == ["k401_2026.year"]
    assert found["k401_2026.year"] | {"source": None} == {
        "target": "facts", "name": "k401_2026.year", "supplied": 2026, "derived": 2024, "source": None,
        "status": "open", "decided_by": None, "decided_at": None, "note": None}


def loaded(tmp_path, rows, name="t"):
    file = tmp_path / f"{name}.dmn"
    file.write_text(dmn(rows), encoding="utf-8")
    return load_table(file)


SHIFTED = [("&gt;= 4", 10, "x"), ("-", 7, "y")]


def test_the_grid_takes_every_bound_points_either_side_scaled_to_the_gap_one_beyond_and_zero(tmp_path):
    grid = derive_check.grid(loaded(tmp_path, GOOD, "a"), loaded(tmp_path, SHIFTED, "b"))
    assert [row["years_of_service"] for row in grid] == [0, 3, 3.5, 4, 4.5, 5, 5.5, 6]


def test_two_tables_that_differ_at_a_bound_are_listed_there(tmp_path):
    first, second = loaded(tmp_path, GOOD, "a"), loaded(tmp_path, SHIFTED, "b")
    assert derive_check.table_differences(first, second, ["carryover_limit"]) == [
        {"inputs": {"years_of_service": years}, "first": {"carryover_limit": 7}, "second": {"carryover_limit": 10}}
        for years in (4, 4.5)]


def test_tables_that_differ_only_in_their_reasons_agree(tmp_path):
    first = loaded(tmp_path, GOOD, "a")
    second = loaded(tmp_path, [("&gt;= 5", 10, "other words"), ("&lt; 5", 7, "more")], "b")
    assert not derive_check.table_differences(first, second, ["carryover_limit"])


def test_two_tables_with_no_row_for_an_input_agree_there(tmp_path):
    first, second = loaded(tmp_path, [("&gt;= 5", 10, "x")], "a"), loaded(tmp_path, [("&gt;= 5", 10, "y")], "b")
    assert not derive_check.table_differences(first, second, ["carryover_limit"])


def test_facts_agree_within_half_a_percent():
    assert derive_check.same_value(6300, 6310) and not derive_check.same_value(6300, 6400)
    assert derive_check.same_value(True, True) and not derive_check.same_value(3, 100)
    assert not derive_check.same_value(True, 1) and derive_check.same_value("a", "a")


def test_fact_agreement_is_symmetric_and_relative_with_no_floor():
    assert not derive_check.same_value(0.035, 0.039) and derive_check.same_value(0.035, 0.035)
    assert derive_check.same_value(100, 99.5) and derive_check.same_value(99.5, 100)
    assert derive_check.same_value(0, 0) and not derive_check.same_value(0, 0.001)
    assert not derive_check.same_value(0.001, 0)


def test_a_disagreement_says_where_and_what_each_reading_gives():
    differences = [{"inputs": {"years_of_service": 4}, "first": {"carryover_limit": 7},
                    "second": {"carryover_limit": 10}}]
    assert derive_check.disagreement("gemma4:26b", differences) == (
        "a second reading by gemma4:26b disagrees: at years_of_service=4, 7 against 10")
    assert derive_check.disagreement("m", [{"first": 100, "second": 3}]) == (
        "a second reading by m disagrees: 100 against 3")


def test_a_long_disagreement_lists_three_points_and_counts_the_rest():
    differences = [{"inputs": {"d": day}, "first": {"allowed": False, "weeks": 0},
                    "second": {"allowed": True, "weeks": 12}} for day in range(5)]
    point = "allowed false, weeks 0 against allowed true, weeks 12"
    assert derive_check.disagreement("m", differences) == (
        f"a second reading by m disagrees: at d=0, {point}; at d=1, {point}; at d=2, {point}; and at 2 more")


def rendered(tmp_path, name, inputs, outputs, rows):
    """A table rendered as clm-derive renders one: rows of ({input: test}, {output: value})."""
    slot = Slot(name, "decisions", "A test rule.", "handbook", inputs, outputs)
    file = tmp_path / f"{name}.dmn"
    file.write_text(derive.render_dmn(slot, [{"inputs": tests, "outputs": values, "reason": "r"}
                                             for tests, values in rows], "hb-leave"), encoding="utf-8")
    return load_table(file)


def numbers_table(tmp_path, name, rows):
    return rendered(tmp_path, name, {"n": "number"}, {"k": "number"}, [({"n": test}, {"k": k}) for test, k in rows])


def test_the_grid_takes_points_between_bounds_less_than_one_apart(tmp_path):
    first = numbers_table(tmp_path, "a", [("< 0.5", 1), ("-", 0)])
    second = numbers_table(tmp_path, "b", [("< 0.7", 1), ("-", 0)])
    assert [item["inputs"]["n"] for item in derive_check.table_differences(first, second, ["k"])] == [0.5, 0.6]


def integers_table(tmp_path, name, rows):
    return rendered(tmp_path, name, {"n": "integer"}, {"k": "number"}, [({"n": test}, {"k": k}) for test, k in rows])


@pytest.mark.parametrize("one, other", [
    ([("<= 17", 1), (">= 18", 2)], [("< 18", 1), (">= 18", 2)]),
    ([("< 2", 1), ("-", 0)], [("[0..2)", 1), ("-", 0)]),
    ([("<= 0", 1), ("-", 0)], [("0", 1), ("-", 0)])])
def test_integer_rules_that_agree_on_every_whole_number_agree(tmp_path, one, other):
    first, second = integers_table(tmp_path, "a", one), integers_table(tmp_path, "b", other)
    assert not derive_check.table_differences(first, second, ["k"], {"n": "integer"})
    assert derive_check.table_differences(first, second, ["k"], {"n": "number"})


def test_the_grid_for_an_integer_takes_whole_numbers_and_negatives_only_beside_a_negative_bound(tmp_path):
    shifted = loaded(tmp_path, SHIFTED, "b")
    assert [row["years_of_service"] for row in derive_check.grid(loaded(tmp_path, GOOD, "a"), shifted,
                                                                  {"years_of_service": "integer"})] == [0, 3, 4, 5, 6]
    below = integers_table(tmp_path, "c", [("< -5", 1), ("-", 0)])
    assert [row["n"] for row in derive_check.grid(below, below, {"n": "integer"})] == [-6, -5, -4, 0]
    half = integers_table(tmp_path, "d", [("< 2.5", 1), ("-", 0)])
    assert [row["n"] for row in derive_check.grid(half, half, {"n": "integer"})] == [0, 1, 2, 3, 4]


def test_the_grid_takes_points_below_zero(tmp_path):
    first = numbers_table(tmp_path, "a", [("< -5", 1), ("-", 0)])
    second = numbers_table(tmp_path, "b", [("< -3", 1), ("-", 0)])
    assert [point["n"] for point in derive_check.grid(first, second)] == [-6, -5, -4, -3, -2, 0]
    assert [item["inputs"]["n"] for item in derive_check.table_differences(first, second, ["k"])] == [-5, -4]


def test_text_outputs_that_differ_only_in_case_or_spaces_agree(tmp_path):
    first = rendered(tmp_path, "a", {"n": "number"}, {"kind": "string"}, [({"n": "-"}, {"kind": "Full Time"})])
    second = rendered(tmp_path, "b", {"n": "number"}, {"kind": "string"}, [({"n": "-"}, {"kind": " full time "})])
    other = rendered(tmp_path, "c", {"n": "number"}, {"kind": "string"}, [({"n": "-"}, {"kind": "part time"})])
    assert not derive_check.table_differences(first, second, ["kind"])
    assert derive_check.table_differences(first, other, ["kind"])[0]["first"] == {"kind": "Full Time"}


def test_the_grid_takes_each_text_value_and_one_neither_table_names(tmp_path):
    rows = [({"plan": '"gold"'}, {"k": 2}), ({"plan": "-"}, {"k": 1})]
    first = rendered(tmp_path, "a", {"plan": "string"}, {"k": "number"}, rows)
    second = rendered(tmp_path, "b", {"plan": "string"}, {"k": "number"}, [({"plan": '"silver"'}, {"k": 2}), rows[1]])
    assert [point["plan"] for point in derive_check.grid(first, second)] == ["gold", "silver", "(none of these)"]
    assert [item["inputs"]["plan"] for item in derive_check.table_differences(first, second, ["k"])] == [
        "gold", "silver"]


def test_the_grid_takes_both_booleans(tmp_path):
    first = rendered(tmp_path, "a", {"member": "boolean"}, {"k": "number"}, [({"member": "true"}, {"k": 2}),
                                                                         ({"member": "-"}, {"k": 1})])
    second = rendered(tmp_path, "b", {"member": "boolean"}, {"k": "number"}, [({"member": "-"}, {"k": 1})])
    assert [point["member"] for point in derive_check.grid(first, second)] == [True, False]
    assert derive_check.table_differences(first, second, ["k"]) == [
        {"inputs": {"member": True}, "first": {"k": 2}, "second": {"k": 1}}]


def test_the_grid_for_two_inputs_takes_every_combination(tmp_path):
    inputs, outputs = {"years": "number", "member": "boolean"}, {"k": "number"}
    first = rendered(tmp_path, "a", inputs, outputs, [({"years": ">= 5", "member": "true"}, {"k": 2}),
                                                   ({"years": "-", "member": "-"}, {"k": 1})])
    second = rendered(tmp_path, "b", inputs, outputs, [({"years": ">= 5", "member": "-"}, {"k": 2}),
                                                    ({"years": "-", "member": "-"}, {"k": 1})])
    grid = derive_check.grid(first, second)
    assert len(grid) == 4 * 2 and {"years": 6, "member": False} in grid
    assert [item["inputs"] for item in derive_check.table_differences(first, second, ["k"])] == [
        {"years": 5, "member": False}, {"years": 6, "member": False}]


def test_a_grid_over_its_limit_is_refused_before_it_is_built(tmp_path, monkeypatch):
    inputs, outputs = {"years": "number", "member": "boolean"}, {"k": "number"}
    first = rendered(tmp_path, "a", inputs, outputs, [({"years": ">= 5", "member": "-"}, {"k": 1})])
    monkeypatch.setattr(derive_check, "MAX_POINTS", 7)
    with pytest.raises(DmnError, match="8 points to compare, more than 7"):
        derive_check.table_differences(first, first, ["k"])


@pytest.mark.parametrize("value", [1500000.0, 123456.78, 1e-7, 1e16, 0.1, 6300])
def test_a_number_output_is_written_whole_without_an_exponent_or_rounding(tmp_path, value):
    table = rendered(tmp_path, "a", {"n": "number"}, {"k": "number"}, [({"n": "-"}, {"k": value})])
    assert table.rules[0].values[0] == value


def test_a_list_output_is_written_as_its_items_joined(tmp_path):
    table = rendered(tmp_path, "a", {"n": "number"}, {"k": "list"}, [({"n": "-"}, {"k": ["full time", "part time"]})])
    assert table.rules[0].values[0] == "full time; part time"


def test_a_failed_or_rejected_fact_opens_no_conflict(env):
    layer = derived.empty("handbook")
    blank = {key: value for key, value in fact(2024, prop="year").items() if key != "value"}
    layer["facts"]["k401_2026.year"] = blank
    assert not derive_check.conflicts(env, layer)
    layer["facts"]["k401_2026.year"] = {**fact(2024, "which for 2024 is", ("(2)",), prop="year"),
                                        "status": "rejected"}
    assert not derive_check.conflicts(env, layer)


def test_each_item_of_a_list_fact_has_its_figures_checked(env):
    tiers = Slot("k401_2026.tiers", "facts", "The match tiers.", "handbook", node="k401_2026", property="tiers",
                 type="list")
    listed = dataclasses.replace(env, slots={**env.slots, tiers.name: tiers})
    entry = {**fact(["100% of the first 3%", "50% up to 6%"], "Meridian matches 100% of the first 3%", ("(3)",),
                    prop="tiers")}
    found = derive_check.check(listed, "facts", tiers.name, entry, on=ON)
    assert "figure 50 (value[1]) is not in the quote" in found and "figure 6 (value[1]) is not in the quote" in found
    assert not [failure for failure in found if "value[0]" in failure]
