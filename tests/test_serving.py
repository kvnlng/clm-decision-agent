"""Tests for serving.py: which tools a server offers, and whether it is read-only, from [serve] and the command
line."""
# Test names say what they check.
# pylint: disable=missing-function-docstring
import pytest

from clm_decision import serving
from clm_decision.scenario import ScenarioError
from clm_decision.serving import Serve

NAMES = ["describe", "query_employee", "query_plan", "scenario_match", "scenario_tuition", "search_handbook",
         "document_handbook", "rules", "rule"]


def test_by_default_every_tool_is_offered_and_none_left_out():
    assert serving.choose(Serve(), NAMES) == (NAMES, [])


def test_tools_offers_only_those_and_describe_always():
    assert serving.choose(Serve(tools=("rules", "rule")), NAMES) == (
        ["describe", "rules", "rule"],
        ["query_employee", "query_plan", "scenario_match", "scenario_tuition", "search_handbook",
         "document_handbook"])


def test_a_name_ending_in_a_star_matches_a_prefix():
    offered, _ = serving.choose(Serve(tools=("scenario_*", "search_*")), NAMES)
    assert offered == ["describe", "scenario_match", "scenario_tuition", "search_handbook"]


def test_exclusion_wins_over_inclusion():
    offered, left_out = serving.choose(Serve(tools=("scenario_*",), exclude_tools=("scenario_tuition",)), NAMES)
    assert offered == ["describe", "scenario_match"] and "scenario_tuition" in left_out


def test_exclude_tools_alone_removes_only_those():
    offered, left_out = serving.choose(Serve(exclude_tools=("query_*",)), NAMES)
    assert left_out == ["query_employee", "query_plan"] and len(offered) == len(NAMES) - 2


def test_a_pattern_that_matches_describe_leaves_it_offered():
    assert serving.choose(Serve(exclude_tools=("*",)), NAMES) == (["describe"], NAMES[1:])


def test_excluding_describe_by_name_stops_with_one_line():
    with pytest.raises(ScenarioError, match=r"^exclude_tools: describe is always offered"):
        serving.choose(Serve(exclude_tools=("describe",)), NAMES)


@pytest.mark.parametrize(("serve", "where", "name"), [
    (Serve(tools=("nope",)), "tools", "'nope'"), (Serve(exclude_tools=("calculation_*",)), "exclude_tools",
                                                  "'calculation_\\*'")])
def test_an_unknown_name_or_a_pattern_matching_nothing_names_it_and_the_tools(serve, where, name):
    with pytest.raises(ScenarioError, match=rf"^{where}: {name} matches no tool; the tools are: describe, "
                                            r"query_employee, .*, rule$"):
        serving.choose(serve, NAMES)


def test_no_serve_table_is_the_default():
    assert serving.from_file({}) == Serve()


def test_the_serve_table_is_read():
    assert serving.from_file({"serve": {"tools": ["describe", "search_*"], "exclude_tools": ["rule"],
                                        "read_only": True}}) == Serve(("describe", "search_*"), ("rule",), True)


@pytest.mark.parametrize(("table", "problem"), [
    ({"tools": "describe"}, r"serve.tools: a list of tool names"),
    ({"exclude_tools": [1]}, r"serve.exclude_tools: a list of tool names"),
    ({"tools": []}, r"serve.tools: a list of tool names"),
    ({"read_only": "yes"}, r"serve.read_only: true or false"),
    ({"toolsets": ["x"]}, r"serve: unknown setting toolsets; only tools, exclude_tools and read_only"),
])
def test_a_bad_serve_table_is_one_line(table, problem):
    with pytest.raises(ScenarioError, match=rf"^{problem}"):
        serving.from_file({"serve": table})


def test_the_command_line_replaces_the_files_setting_never_adds():
    file = Serve(tools=("rules",), exclude_tools=("rule",))
    assert serving.merged(file, tools="search_*", exclude_tools=None, read_only=False) == \
        Serve(("search_*",), ("rule",), False)
    assert serving.merged(file, tools=None, exclude_tools="query_*,scenario_*", read_only=True) == \
        Serve(("rules",), ("query_*", "scenario_*"), True)


def test_read_only_in_the_file_stays_without_the_flag():
    assert serving.merged(Serve(read_only=True), None, None, False).read_only


def test_an_empty_flag_names_no_tool():
    with pytest.raises(ScenarioError, match=r"^--tools: names no tool"):
        serving.merged(Serve(), tools=" , ", exclude_tools=None, read_only=False)
