"""Tests for the CrowdStrike check_call story: an agent proposes both a rebooking and a refund."""
# Test names say what they check.
# pylint: disable=missing-function-docstring,missing-class-docstring
from conftest import FakeRouter
from test_crowdstrike_tools import SCENARIO

from check_calls import PROPOSALS, check, report
from scenario import load_scenario


class RefundMinded(FakeRouter):
    """Approves a refund for a message that mentions money, and a hold for any other."""

    def check_tool(self, state_text, scenario):
        self.tool_states.append(state_text)
        refund = "issue_refund" in state_text
        return 0.8 if refund == ("money" in state_text) else 0.3


def test_each_case_gets_both_proposals_and_is_caught_only_when_the_wrong_one_is_refused():
    cases = [{"text": "Rebook me please", "context": {"booking": "B1", "now": 0}, "route": "Rebook"},
             {"text": "Just give me my money back", "context": {"booking": "B2", "now": 0}, "route": "Refund"},
             {"text": "I want my money back or a seat", "context": {"booking": "B3", "now": 0}, "route": "Rebook"},
             {"text": "Fine, I'll wait", "route": "No_Action"}]
    router = RefundMinded({"route": "Rebook", "confidence": 0.9, "probabilities": {}, "answers": {}})
    found = check(cases, load_scenario(SCENARIO), router)
    assert len(router.tool_states) == 6 and all('"booking": "B' in state for state in router.tool_states)
    assert found["approved"] == {"Rebook": {"hold_seats": 1, "issue_refund": 1},
                                 "Refund": {"hold_seats": 0, "issue_refund": 1}}
    assert found["cases"] == {"Rebook": 2, "Refund": 1}
    assert found["caught"] == {"Rebook": 1, "Refund": 1}
    assert found["preferred"] == {"Rebook": 1, "Refund": 1}         # the right call had the higher probability
    assert "Rebook" in report(found) and "caught 1 of 2" in report(found) and "preferred 1 of 2" in report(found)
    assert set(PROPOSALS) == {"hold_seats", "issue_refund"}
