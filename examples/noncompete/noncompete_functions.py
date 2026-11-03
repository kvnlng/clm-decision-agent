"""The non-compete example's queries, calculations and branches (environment.toml).

The company, Meridian Logistics, is made up. Retrieval finds evidence; only the decision tables decide: no rule here
takes a search score as an input. Not legal advice.
"""
NOTICE = ("Not legal advice: a starting point for counsel, from the statutes and court opinions cited. Check each "
          "citation before relying on it.")
CANNOT_TELL = ("This cannot tell you how a particular judge will rule, facts outside the agreement, law after the "
               "collection was built, or what the incomplete citation graph misses.")
YES = ("yes", "true")


def _node(graph, ident, kind):
    node = graph.node(ident) if ident in {n["id"] for n in graph.nodes(kind)} else None
    if node is None:
        raise LookupError(f"no {kind} {ident}")
    return node


def agreement(graph, agreement):  # pylint: disable=redefined-outer-name  # the query names its parameter
    """The agreement's terms."""
    node = _node(graph, agreement, "agreement")
    return [{"agreement": agreement, "length_months": node["length_months"], "area": node["area"],
             "garden_leave": node["garden_leave"] in YES, "notice_days": node["notice_days"],
             "separate_notice": node["separate_notice"] in YES, "choice_of_law": node["choice_of_law"],
             "signed": node["signed"], "clause": node["clause"]}]


def workforce(graph, state):
    """The staff bound by the agreement in a state."""
    _node(graph, state, "state")
    rows = [{"state": state, "employees": node["employees"], "lowest_pay": node["lowest_pay"],
             "all_exempt": node["all_exempt"] in YES}
            for node in graph.neighbours(state, "works_in", "in")]
    if not rows:
        raise LookupError(f"no staff in {state}")
    return rows


def pay_threshold(graph, state):
    """The state's pay threshold for a non-compete, with its year and source; amount 0 where there is none."""
    _node(graph, state, "state")
    rows = [{"state": state, "amount": node["amount"], "year": node["year"], "source": node["source"]}
            for node in graph.neighbours(state, "applies_in", "in")]
    return rows or [{"state": state, "amount": 0, "year": None, "source": "no pay threshold in this example's law"}]


def employee(graph, ident):
    """Where an employee resides and works, and the agreement they signed."""
    node = _node(graph, ident, "employee")
    bound = [n["id"] for n in graph.neighbours(ident, "bound_by", "out")]
    return [{"employee": ident, "resides": node["resides"], "works": node["works"], "agreement": bound[0]}]


def pay_test(lowest_pay, threshold):
    """Whether the lowest-paid bound employee earns below the threshold, and by how much."""
    below = threshold > 0 and lowest_pay <= threshold
    return {"below_threshold": below, "shortfall": round(threshold - lowest_pay, 2) if below else 0}


def court_record(cases, length_months, nationwide):
    """Of the similar cases restricting at least this much, how many courts enforced as written."""
    as_broad = [case for case in cases if case["length_months"] >= length_months
                and (case["nationwide"] or not nationwide)]
    enforced = sum(case["outcome"] == "enforced" for case in as_broad)
    narrowed = sum(case["outcome"] in ("reformed", "void") for case in as_broad)
    lengths = sorted(case["length_months"] for case in cases if case["outcome"] == "enforced")
    return {"similar": len(cases), "as_broad": len(as_broad), "enforced_as_written": enforced,
            "narrowed_or_voided": narrowed, "share_enforced": round(enforced / len(as_broad), 2) if as_broad else 0,
            "longest_enforced_months": lengths[-1] if lengths else 0}


def breadth(length_months, nationwide):
    """How broad the terms are: months, doubled when nationwide."""
    return {"breadth": length_months * (2 if nationwide else 1)}


def by_state(steps, params, levers):
    """Question 1: whether the agreement holds in one state, and what must change."""
    state = levers["state"]
    [terms] = steps.query("agreement", agreement=params["agreement"])
    staff = steps.query("workforce", state=state)
    [threshold] = steps.query("pay_threshold", state=state)
    lowest = min(row["lowest_pay"] for row in staff)
    test = steps.calculate("pay_test", lowest_pay=lowest, threshold=threshold["amount"])
    verdict = steps.decide("state_law", state=state, below_threshold=test["below_threshold"],
                           all_exempt=all(row["all_exempt"] for row in staff), notice_days=terms["notice_days"],
                           garden_leave=terms["garden_leave"], length_months=terms["length_months"],
                           signed_after_ban=terms["signed"] >= "2023-07-01", separate_notice=terms["separate_notice"])
    return {"employees": sum(row["employees"] for row in staff), "holds": verdict["holds"],
            "what_must_change": verdict["reason"], "citation": verdict["citation"], "notice": NOTICE}


def _nationwide(area):
    return any(word in (area or "").lower() for word in ("nation", "united states", "country", "anywhere"))


def _still_good_law(steps, cases):
    """For the three weightiest cases, read up to three passages that cite each and judge whether they depart."""
    for case in sorted(cases, key=lambda c: c["weight"] != "binding")[:3]:
        departures = []
        for citer in steps.cited_by("cases", case["document"], 1)[:3]:
            if citer["passage"]:
                text = steps.document("cases", citer["document"], citer["passage"])["passages"][0]["text"]
                departures.append(steps.judge("departs", text)["probability"])
        case["citing_cases_read"] = len(departures)
        case["most_critical_citing"] = max(departures, default=0.0)


def narrow(steps, params, levers):  # pylint: disable=too-many-locals
    """Question 2: whether courts in the state would enforce terms this broad, from similar cases."""
    length, nationwide = levers["length_months"], levers["area"] == "nationwide"
    where = "anywhere in the United States" if nationwide else f"in {params['served']}"
    rows = steps.search("cases", f"{params['role']} covenant not to compete for {length} months {where}",
                        {"state": params["state"]}, 50)
    cases, seen = [], set()
    for row in rows:
        meta = row["metadata"]
        if row["document"] in seen:
            continue
        seen.add(row["document"])
        weight = steps.decide("court_weight", court_level=meta.get("court_level", "other"))["weight"]
        known = isinstance(meta.get("length_months"), (int, float)) and meta.get("outcome") in (
            "enforced", "reformed", "void")
        if steps.decide("similar_case", weight=weight, facts_known=known)["similar"]:
            cases.append({"document": row["document"], "citation": row["citation"], "title": row["title"],
                          "weight": weight, "role": meta.get("role"), "length_months": meta["length_months"],
                          "area": meta.get("area"), "nationwide": _nationwide(meta.get("area")),
                          "outcome": meta["outcome"], "reformed_to": meta.get("reformed_to")})
    record = steps.calculate("court_record", cases=cases, length_months=length, nationwide=nationwide)
    steps.decide("likely_enforced", as_broad=record["as_broad"], share_enforced=record["share_enforced"])
    terms = f"A {params['role']} may not work for a competitor for {length} months after leaving, {where}."
    broader = steps.judge("broader_than_needed", terms)["probability"]
    _still_good_law(steps, cases)
    return {**steps.calculate("breadth", length_months=length, nationwide=nationwide), **record,
            "broader_than_needed": broader, "cases": cases, "notice": NOTICE, "cannot_tell": CANNOT_TELL}
