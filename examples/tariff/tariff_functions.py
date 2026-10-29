"""The tariff example's owner-written functions: the named queries on the built-in graph, the calculations, and the
sourcing scenario's branch.

The branch asks for everything through `steps`, so each query, rule and calculation is recorded with its numbers. It
moves rows between steps but does no arithmetic of its own: the calculations do that, and the rules judge.
"""


def _region(graph, supplier: str) -> str:
    return graph.neighbours(supplier, "located_in")[0]["name"]


def product_facts(graph, product):
    """The product's name, Q3 units and unit price."""
    node = graph.node(product)
    return [{"id": node["id"], "name": node["name"], "q3_units": node["q3_units"], "unit_price": node["unit_price"]}]


def components_of(graph, product):
    """The components the product is built from."""
    return [{"id": edge["source"], "name": graph.node(edge["source"])["name"],
             "category": graph.node(edge["source"])["category"], "units_per_product": edge["units_per_product"]}
            for edge in graph.edges(relation="part_of", target=product)]


def suppliers_of(graph, component):
    """Who supplies the component, and on what terms."""
    rows = []
    for edge in graph.edges(relation="supplies", target=component):
        supplier = graph.node(edge["source"])
        rows.append({"id": supplier["id"], "name": supplier["name"], "region": _region(graph, supplier["id"]),
                     "unit_price": edge["unit_price"], "lead_time_days": supplier["lead_time_days"],
                     "defect_rate": supplier["defect_rate"], "freight_per_unit": supplier["freight_per_unit"]})
    return rows


def key_clients_of(graph, product):
    """The key clients buying the product."""
    clients = [graph.node(edge["source"]) for edge in graph.edges(relation="buys", target=product)
               if edge.get("key") == "yes"]
    return [{"id": client["id"], "name": client["name"], "tier": client["tier"]} for client in clients]


def supply_dependencies(graph, suppliers):
    """Each supplier's raw-material providers, up to three steps away."""
    return [{"supplier": supplier, "provider": node["id"], "provider_name": node["name"], "path": node["path"]}
            for supplier in suppliers for node in graph.reachable(supplier, ["depends_on"], 3)
            if node["kind"] == "provider"]


def single_points_of_failure(graph, suppliers):
    """The providers every one of the suppliers depends on."""
    return [{"id": node["id"], "name": node["name"], "kind": node["kind"],
             "via": [{"supplier": supplier, "path": path} for supplier, path in node["via"].items()]}
            for node in graph.single_points_of_failure(suppliers, ["depends_on"], 3) if node["kind"] == "provider"]


def _weights(offers: list, region_b_share: float) -> dict[str, list[tuple[dict, float]]]:
    """Component -> its offers, each with the share of the component's volume it carries."""
    regions = {"Region A": 1 - region_b_share, "Region B": region_b_share}
    by_component: dict[str, list[tuple[dict, float]]] = {}
    for offer in offers:
        same = [other for other in offers if other["component"] == offer["component"]
                and other["region"] == offer["region"]]
        by_component.setdefault(offer["component"], []).append((offer, regions[offer["region"]] / len(same)))
    return by_component


def branch_figures(q3_units, unit_price, offers, region_b_share):  # pylint: disable=too-many-locals
    """The branch's Q3 money, delivery and quality figures. Lead time and defect rate are volume-weighted per component;
    the product waits for its slowest component, and the extension is measured against sourcing all in Region A."""
    cost = tariff = 0.0
    lead_times, baseline, defects = [], [], []
    for component, weighted in _weights(offers, region_b_share).items():
        for offer, weight in weighted:
            units = q3_units * offer["units_per_product"] * weight
            tariff += units * offer["unit_price"] * offer["tariff_rate"]
            cost += units * (offer["unit_price"] * (1 + offer["tariff_rate"]) + offer["freight_per_unit"])
        lead_times.append(sum(offer["lead_time_days"] * weight for offer, weight in weighted))
        defects.append(sum(offer["defect_rate"] * weight for offer, weight in weighted))
        baseline.append(max(offer["lead_time_days"] for offer in offers
                            if offer["component"] == component and offer["region"] == "Region A"))
    revenue = q3_units * unit_price
    margin = revenue - cost
    lead_time = max(lead_times)
    return {"q3_revenue": round(revenue, 2), "q3_cost": round(cost, 2), "q3_tariff_cost": round(tariff, 2),
            "q3_margin": round(margin, 2), "q3_margin_rate": round(margin / revenue, 4),
            "lead_time_days": round(lead_time, 2), "delivery_extension_days": round(lead_time - max(baseline), 2),
            "defect_rate": round(sum(defects) / len(defects), 4)}


def concentration(dependencies, offers, region_b_share):
    """The provider carrying the largest share of supply: for each component, the share of its volume bought from
    suppliers that depend on the provider, averaged over the components."""
    by_component = _weights(offers, region_b_share)
    shares = {}
    for provider in {row["provider"] for row in dependencies}:
        behind = {row["supplier"] for row in dependencies if row["provider"] == provider}
        shares[provider] = sum(sum(weight for offer, weight in weighted if offer["supplier"] in behind)
                               for weighted in by_component.values()) / len(by_component)
    provider = max(sorted(shares), key=lambda item: shares[item])
    name = next(row["provider_name"] for row in dependencies if row["provider"] == provider)
    return {"largest_share": round(shares[provider], 4), "provider": name, "provider_id": provider}


def sourcing_branch(steps, params, levers):
    """One branch of the sourcing scenario: every limit is checked, every number comes from a step."""
    share = levers["region_b_share"]
    product = steps.query("product_facts", product=params["product"])[0]
    offers = []
    for component in steps.query("components_of", product=params["product"]):
        for supplier in steps.query("suppliers_of", component=component["id"]):
            rate = steps.decide("tariff_rate", region=supplier["region"], category=component["category"])["rate"]
            offers.append({"component": component["id"], "supplier": supplier["id"], "region": supplier["region"],
                           "unit_price": supplier["unit_price"], "tariff_rate": rate,
                           "freight_per_unit": supplier["freight_per_unit"],
                           "lead_time_days": supplier["lead_time_days"], "defect_rate": supplier["defect_rate"],
                           "units_per_product": component["units_per_product"]})
    figures = steps.calculate("branch_figures", q3_units=product["q3_units"], unit_price=product["unit_price"],
                              offers=offers, region_b_share=share)
    for client in steps.query("key_clients_of", product=params["product"]):
        steps.decide("sla_threshold", tier=client["tier"], extension_days=figures["delivery_extension_days"])
    steps.decide("quality_threshold", defect_rate=figures["defect_rate"])
    # Only the suppliers this branch buys from count towards a provider's share
    in_use = sorted({offer["supplier"] for offer in offers
                     if (offer["region"] == "Region B" and share > 0) or (offer["region"] == "Region A" and share < 1)})
    dependencies = steps.query("supply_dependencies", suppliers=in_use)
    spread = steps.calculate("concentration", dependencies=dependencies, offers=offers, region_b_share=share)
    steps.decide("supply_concentration", share=spread["largest_share"], provider=spread["provider"])
    return figures
