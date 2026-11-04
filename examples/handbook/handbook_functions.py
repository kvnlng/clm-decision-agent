"""The handbook example's queries, calculations and branches (environment.toml).

Meridian Logistics, its employees and its handbook are made up. The rules decide and the calculations count; the
handbook's text is what the rules cite. Not HR, tax or legal advice.
"""
from datetime import date

NOTICE = ("Not HR, tax or legal advice: Meridian Logistics, its people and its handbook are made up for this "
          "example.")


def _node(graph, ident, kind):
    if ident not in {n["id"] for n in graph.nodes(kind)}:
        raise LookupError(f"no {kind} {ident}")
    return graph.node(ident)


def _day(text, what):
    try:
        return date.fromisoformat(str(text))
    except ValueError as error:
        raise ValueError(f"{what} must be a date such as 2026-12-01, not {text!r}") from error


def employee(graph, employee, as_of):  # pylint: disable=redefined-outer-name  # the query names its parameter
    """The employee's record as of a date: days employed counting the first day as day 1, and full years of
    service."""
    node = _node(graph, employee, "employee")
    hired, today = _day(node["hired"], "hired"), _day(as_of, "as_of")
    if today < hired:
        raise ValueError(f"{employee} was hired on {hired}, after {today}")
    years = today.year - hired.year - ((today.month, today.day) < (hired.month, hired.day))
    return [{"employee": employee, "state": node["state"], "office": node["office"], "hired": str(hired),
             "as_of": str(today), "days_employed": (today - hired).days + 1, "years_of_service": years,
             "salary": node["salary"], "contribution_percent": node["contribution_percent"],
             "leave_balance_days": node["leave_balance_days"]}]


def plan(graph):
    """The 401(k) plan's IRS limit on employee contributions and its cap on the company match, with the source."""
    node = _node(graph, "k401_2026", "plan")
    return [{"year": node["year"], "irs_limit": node["irs_limit"], "match_cap": node["match_cap"],
             "source": node["source"]}]


def company_match(salary, contribution_percent, match_cap):
    """100% of the first 3% of pay contributed and 50% of the next 3%, capped (handbook § 4(3) and (4))."""
    first = min(contribution_percent, 3)
    second = min(max(contribution_percent - 3, 0), 3)
    return {"match": round(min(salary * (first + second / 2) / 100, match_cap), 2)}


def carryover_days(balance, days_taken, limit):
    """The days left after taking some, how many carry over, and how many are forfeited."""
    left = max(balance - days_taken, 0)
    return {"days_left": left, "carried_days": min(left, limit), "forfeited_days": max(left - limit, 0)}


def leave_pay(salary, full_pay_weeks, partial_pay_weeks, partial_pay_percent, weeks_taken):
    """Weekly pay at full and reduced pay, and the total for the weeks taken."""
    weekly = round(salary / 52, 2)
    partial = round(weekly * partial_pay_percent / 100, 2)
    full = min(weeks_taken, full_pay_weeks)
    reduced = min(max(weeks_taken - full_pay_weeks, 0), partial_pay_weeks)
    return {"weekly_pay": weekly, "partial_weekly_pay": partial,
            "total_pay": round(full * weekly + reduced * partial, 2)}


def reimbursement(tuition, percent, annual_cap):  # pylint: disable=redefined-outer-name  # the calculation names it
    """The share of tuition repaid, up to the yearly cap."""
    return {"reimbursed": round(min(tuition * percent / 100, annual_cap), 2)}


def match(steps, params, levers):
    """Question 1: the company match now and at a contribution rate, the most it can be, the cap and the IRS
    limit."""
    [person] = steps.query("employee", employee=params["employee"], as_of=params["as_of"])
    [terms] = steps.query("plan")
    steps.decide("match_eligible", days_employed=person["days_employed"])
    vested = steps.decide("vesting", years_of_service=person["years_of_service"])
    at_rate = steps.calculate("company_match", salary=person["salary"],
                              contribution_percent=levers["contribution_percent"], match_cap=terms["match_cap"])
    now = steps.calculate("company_match", salary=person["salary"],
                          contribution_percent=person["contribution_percent"], match_cap=terms["match_cap"])
    most = steps.calculate("company_match", salary=person["salary"], contribution_percent=6,
                           match_cap=terms["match_cap"])
    return {"match_at_rate": at_rate["match"], "current_percent": person["contribution_percent"],
            "company_match": now["match"], "most_match": most["match"], "match_cap": terms["match_cap"],
            "irs_limit": terms["irs_limit"], "vested_percent": vested["vested_percent"], "notice": NOTICE}


def carryover(steps, params, levers):
    """Question 2: how many unused days carry into next year, and how many are forfeited."""
    [person] = steps.query("employee", employee=params["employee"], as_of=params["as_of"])
    limit = steps.decide("carryover_limit", years_of_service=person["years_of_service"])
    days = steps.calculate("carryover_days", balance=person["leave_balance_days"], days_taken=levers["days_taken"],
                           limit=limit["carryover_limit"])
    return {"balance_days": person["leave_balance_days"], "carryover_limit": limit["carryover_limit"], **days,
            "notice": NOTICE}


def parental_leave(steps, params, levers):
    """Question 3: whether the employee is eligible to start parental leave on the date, for how long, and the
    pay."""
    [person] = steps.query("employee", employee=params["employee"], as_of=params["as_of"])
    terms = steps.decide("parental_eligible", days_employed=person["days_employed"])
    pay = steps.calculate("leave_pay", salary=person["salary"], full_pay_weeks=terms["full_pay_weeks"],
                          partial_pay_weeks=terms["partial_pay_weeks"],
                          partial_pay_percent=terms["partial_pay_percent"], weeks_taken=levers["weeks_taken"])
    return {"full_pay_weeks": terms["full_pay_weeks"], "partial_pay_weeks": terms["partial_pay_weeks"],
            "partial_pay_percent": terms["partial_pay_percent"], **pay, "notice": NOTICE}


def tuition(steps, params, levers):  # pylint: disable=unused-argument  # one calendar year, as the cap is yearly
    """Question 4: whether a course's tuition is repaid, how much, and the yearly cap."""
    [person] = steps.query("employee", employee=params["employee"], as_of=params["as_of"])
    terms = steps.decide("tuition_eligible", years_of_service=person["years_of_service"],
                         job_related=params["job_related"])
    repaid = steps.calculate("reimbursement", tuition=params["tuition"], percent=terms["reimbursed_percent"],
                             annual_cap=terms["annual_cap"])
    return {"tuition": params["tuition"], "reimbursed_percent": terms["reimbursed_percent"],
            "annual_cap": terms["annual_cap"], **repaid, "notice": NOTICE}
