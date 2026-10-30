"""The new-site example's functions: the named queries on the built-in graph and the census snapshot, the
calculations, and the scenario's branch.

A query fails when what it was asked about does not exist. The household income query is an estimate: each row
carries low, central and high, its source and its date, so the engine can try the low and high and search the
headroom. The branch moves rows between steps and does no arithmetic of its own.
"""
import csv
from pathlib import Path

CENSUS = Path(__file__).parent / "data" / "acs_income.csv"


def _node(graph, node_id, kind):
    """The node of that kind; an unknown id is an error naming it."""
    found = {node["id"]: node for node in graph.nodes(kind)}
    if node_id not in found:
        raise LookupError(f"no {kind} {node_id}")
    return found[node_id]


def company_finances(graph):
    """Reserves, net debt, EBITDA and the debt-financed share of an investment."""
    company = _node(graph, "cedar", "company")
    return [{key: company[key] for key in ("name", "reserves", "net_debt", "ebitda", "debt_share")}]


def profit_history(graph):
    """Profit by year."""
    company = _node(graph, "cedar", "company")
    return [{"year": int(key.split("_")[1]), "profit": value} for key, value in sorted(company.items())
            if key.startswith("profit_")]


def planning_assumptions(graph):
    """Finance's planning assumptions."""
    plan = _node(graph, "plan", "policy")
    return [{key: plan[key] for key in ("spend_share", "gross_margin", "revenue_shortfall")}]


def site_options(graph, size):
    """A site size's area, costs and expected capture."""
    site = _node(graph, size, "size")
    return [{"size": size, **{key: site[key] for key in ("sqft", "fit_out", "wages", "capture")}}]


def neighbourhood_facts(graph, neighbourhood):
    """A neighbourhood's households and rent."""
    place = _node(graph, neighbourhood, "neighbourhood")
    return [{"neighbourhood": neighbourhood, "name": place["name"], "households": place["households"],
             "rent_per_sqft": place["rent_per_sqft"]}]


def household_income(graph, neighbourhood):
    """The census estimate of median household income, with its 90% margin of error as low and high."""
    tract = str(_node(graph, neighbourhood, "neighbourhood")["census_tract"]).zfill(6)
    with CENSUS.open(encoding="utf-8") as file:
        row = next((row for row in csv.DictReader(file) if row["tract"] == tract), None)
    if row is None:
        raise LookupError(f"no census figures for tract {tract}")
    estimate, margin = int(row["median_income"]), int(row["margin_of_error"])
    return [{"neighbourhood": neighbourhood, "low": estimate - margin, "central": estimate,
             "high": estimate + margin, "source": f"{row['survey']}, table {row['table']}, {row['name']}",
             "as_of": row["survey"].split()[1]}]


def site_figures(households, income, rent_per_sqft, site, assumptions):
    """Revenue is the households' spending at bakery cafes that the site captures; operating profit is gross profit
    less rent and wages; payback is the fit-out over operating profit, and again with revenue cut by the shortfall."""
    terms, plan = site[0], assumptions[0]
    revenue = households * income * plan["spend_share"] * terms["capture"]
    fixed = rent_per_sqft * terms["sqft"] + terms["wages"]

    def payback(sales):
        profit = sales * plan["gross_margin"] - fixed
        return round(terms["fit_out"] / profit, 2) if profit > 0 else 99.0
    operating = revenue * plan["gross_margin"] - fixed
    return {"revenue": round(revenue), "gross_profit": round(revenue * plan["gross_margin"]),
            "operating_profit": round(operating), "investment": terms["fit_out"], "payback_years": payback(revenue),
            "stressed_payback_years": payback(revenue * (1 - plan["revenue_shortfall"]))}


def financing(investment, finances):
    """The share of reserves used and net debt to EBITDA after borrowing the debt-financed share."""
    company = finances[0]
    borrowed = investment * company["debt_share"]
    return {"from_reserves": round(investment - borrowed), "borrowed": round(borrowed),
            "share_of_reserves": round((investment - borrowed) / company["reserves"], 4),
            "debt_to_ebitda": round((company["net_debt"] + borrowed) / company["ebitda"], 3)}


def profitable_years(history):
    """The number of recorded years with a profit."""
    return {"profitable_years": sum(1 for row in history if row["profit"] > 0)}


def new_site_branch(steps, params, levers):
    """One size of site in the neighbourhood: its figures, how it is financed, every rule and the approval."""
    place = steps.query("neighbourhood_facts", neighbourhood=params["neighbourhood"])[0]
    income = steps.query("household_income", neighbourhood=params["neighbourhood"])[0]
    site = steps.query("site_options", size=levers["size"])
    figures = steps.calculate("site_figures", households=place["households"], income=income["value"],
                              rent_per_sqft=place["rent_per_sqft"], site=site,
                              assumptions=steps.query("planning_assumptions"))
    money = steps.calculate("financing", investment=figures["investment"], finances=steps.query("company_finances"))
    record = steps.calculate("profitable_years", history=steps.query("profit_history"))
    steps.decide("payback", payback_years=figures["payback_years"])
    steps.decide("stress_test", stressed_payback_years=figures["stressed_payback_years"])
    steps.decide("debt_covenant", debt_to_ebitda=money["debt_to_ebitda"])
    steps.decide("reserve_policy", profitable_years=record["profitable_years"],
                 share_of_reserves=money["share_of_reserves"])
    steps.decide("capital_approval", investment=figures["investment"])
    return {**figures, **money}
