"""Tests for rules an agent can explore: rule metadata on decisions, [information.*], and the rules and rule tools."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
from pathlib import Path

import pytest
from conftest import ENV_TOML
from mcp import Client

from clm_decision.environment import load_environment
from clm_decision.environment_server import build_environment_server
from clm_decision.scenario import ScenarioError

TARIFF = Path(__file__).parent.parent / "examples" / "tariff" / "environment.toml"

RULES_TOML = ENV_TOML + '''
[decisions.share_limit.rule]
statement = "At most 90% of a product's parts may come from Region B."
reason = "One region's disruption must not stop production."
owner = "Supply chain"
version = "2026-01"
governs = ["choose_supplier"]

[decisions.payback]
description = "Whether an investment pays back in time"
dmn = "rules/payback.dmn"

[decisions.payback.rule]
statement = "A new site must pay back within 4 years."
reason = "Capital is limited; slower projects crowd out better ones."
owner = "Finance"
version = "2026-03"
governs = ["open_site"]
depends_on = ["reserve_policy"]
needs = ["site_size", "annual_profit"]

[decisions.reserve_policy]
description = "Whether reserves may fund an expansion"
dmn = "rules/reserve_policy.dmn"

[decisions.reserve_policy.rule]
statement = "After two years of profit, up to half the reserves may fund expansion."
reason = "Keep a cushion for a bad year."
owner = "Finance"
version = "2025-11"
needs = ["annual_profit"]

[information.annual_profit]
description = "Profit for each of the last two years, in dollars"
kind = "fact"
query = "suppliers"

[information.site_size]
description = "The size of the site: small or large"
kind = "supplied"
'''


def one_input_dmn(name: str, item: str, refused: str) -> str:
    rule = ('<rule id="{id}"><inputEntry><text>{test}</text></inputEntry><outputEntry><text>{allowed}</text>'
            '</outputEntry><outputEntry><text>"{reason}"</text></outputEntry></rule>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" id="d" name="d" namespace="t">'
            f'<decision id="{name}" name="{name}"><decisionTable id="t" hitPolicy="FIRST">'
            f'<input id="i"><inputExpression typeRef="number"><text>{item}</text></inputExpression></input>'
            '<output id="o1" name="allowed" typeRef="boolean"/><output id="o2" name="reason" typeRef="string"/>'
            + rule.format(id="no", test=refused, allowed="false", reason="Refused")
            + rule.format(id="yes", test="-", allowed="true", reason="Allowed")
            + "</decisionTable></decision></definitions>")


@pytest.fixture
def write_rules(write_environment):
    def _write(toml=RULES_TOML):
        path = write_environment(toml)
        (path.parent / "rules" / "payback.dmn").write_text(one_input_dmn("payback", "payback_years", "&gt; 4"))
        (path.parent / "rules" / "reserve_policy.dmn").write_text(
            one_input_dmn("reserve_policy", "share_of_reserves", "&gt; 0.5"))
        return path
    return _write


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def call(path, tool, arguments):
    async with Client(build_environment_server(load_environment(path))) as client:
        return await client.call_tool(tool, arguments)


@pytest.mark.anyio
async def test_rules_for_a_topic_lists_dependencies_first(write_rules):
    result = (await call(write_rules(), "rules", {"topic": "open_site"})).structured_content
    assert [rule["name"] for rule in result["rules"]] == ["reserve_policy", "payback"]
    payback = result["rules"][1]
    assert payback["statement"] == "A new site must pay back within 4 years." and payback["owner"] == "Finance"
    assert payback["reason"].startswith("Capital is limited") and payback["version"] == "2026-03"
    assert [item["name"] for item in result["information"]] == ["annual_profit", "site_size"]
    assert [item["name"] for item in result["to_ask"]] == ["site_size"]


@pytest.mark.anyio
async def test_rule_shows_its_table_rows(write_rules):
    result = (await call(write_rules(), "rule", {"name": "payback"})).structured_content
    assert result["rows"][0] == {"id": "no", "when": {"payback_years": "> 4"},
                                 "then": {"allowed": False, "reason": "Refused"}}
    assert result["params"] == {"payback_years": "number"} and result["depends_on"] == ["reserve_policy"]


@pytest.mark.anyio
async def test_unknown_topic_is_a_tool_error_listing_topics(write_rules):
    result = await call(write_rules(), "rules", {"topic": "close_site"})
    assert result.is_error and "choose_supplier, open_site" in result.content[0].text


@pytest.mark.anyio
async def test_describe_lists_the_topics(write_rules):
    result = (await call(write_rules(), "describe", {})).structured_content
    assert result["topics"] == ["choose_supplier", "open_site"]


@pytest.mark.parametrize("old, new, setting", [
    ('needs = ["site_size", "annual_profit"]', 'needs = ["site_area"]', "decisions.payback.rule.needs"),
    ('depends_on = ["reserve_policy"]', 'depends_on = ["cash_rule"]', "decisions.payback.rule.depends_on"),
    ('kind = "supplied"', 'kind = "supplied"\nquery = "suppliers"', "information.site_size.query"),
    ('kind = "fact"', 'kind = "estimate"', "information.annual_profit.kind"),
    ('kind = "fact"', 'kind = "rumour"', "information.annual_profit.kind"),
    ('owner = "Finance"\nversion = "2026-03"', 'version = "2026-03"', "decisions.payback.rule.owner"),
])
def test_a_wrong_rule_or_information_is_refused_naming_it(write_rules, old, new, setting):
    assert old in RULES_TOML
    with pytest.raises(ScenarioError, match=f"^{setting}"):
        load_environment(write_rules(RULES_TOML.replace(old, new, 1)))


def test_a_dependency_cycle_is_refused(write_rules):
    toml = RULES_TOML.replace('version = "2025-11"', 'version = "2025-11"\ndepends_on = ["payback"]')
    with pytest.raises(ScenarioError, match=r"^decisions\.\w+\.rule\.depends_on: .*cycle"):
        load_environment(write_rules(toml))


@pytest.mark.anyio
async def test_an_environment_without_rules_offers_no_rules_tools():
    async with Client(build_environment_server(load_environment(TARIFF))) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        described = (await client.call_tool("describe", {})).structured_content
    assert "rules" not in names and "rule" not in names and "topics" not in described
