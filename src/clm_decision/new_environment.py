"""Start a new environment: a folder with a small working environment, to edit into your own.

    uv run clm-new-environment FOLDER

The folder holds a made-up team choosing where to hold an offsite:

    environment.toml            the parts: three named queries, two limits, a calculation and a scenario
    graph/nodes.csv, edges.csv  the team's people by city, the venues and the fares between cities
    rules/capacity.dmn          a limit: everyone must fit
    rules/budget.dmn            a limit: the total cost must be within budget
    <folder>_functions.py       the queries on the graph, the calculation and the scenario's branch

Then `clm-call FOLDER/environment.toml` lists its tools and calls them, and `clm-server` serves them.
docs/environment.md explains every part.
"""
import argparse
import re
import sys
from pathlib import Path

ENVIRONMENT = '''\
# A made-up team choosing where to hold an offsite. Edit it into your own; docs/environment.md explains every part.
# The owner writes every query, rule, calculation and lever here; an agent can only choose among them.

[environment]
name = "Team offsite"
# What an agent reads first: what this is for, which ids to ask about, and where to start
description = """Where a team should hold an offsite. The team to ask about is the Platform team (id T1). Run \\
scenario_offsite to compare the three venues: it reports each one's travel, venue and catering cost, whether \\
everyone fits, and whether it is within the offsite budget."""
functions = "{module}"

[graph]
backend = "builtin"
nodes = "graph/nodes.csv"
edges = "graph/edges.csv"

# Each part's description is what the agent reads: say what the rows hold and in which units
[queries.team_members]
description = "Where a team's people are based: one row per city, with the city's id and name and the headcount."
params = {{ team = "string" }}
function = "team_members"

[queries.venue_facts]
description = "A venue's name, city, capacity in people, day rate in pounds and catering per head per day in pounds."
params = {{ venue = "string" }}
function = "venue_facts"

[queries.return_fare]
description = "The return train fare in pounds per person from one city to another (city ids)."
params = {{ origin = "string", destination = "string" }}
function = "return_fare"

# A limit is a decision whose outputs include allowed and reason
[decisions.capacity]
description = "Whether everyone fits: allowed when the spare seats are zero or more."
dmn = "rules/capacity.dmn"

[decisions.budget]
description = "Whether an offsite's total cost in pounds is within the offsite budget of 9,000 pounds."
dmn = "rules/budget.dmn"

# Arithmetic belongs in a calculation, so every figure in an answer has a step behind it
[calculations.offsite_figures]
description = """The offsite's travel, venue, catering and total cost in pounds, the headcount and the spare seats, \\
from the team's cities, each city's return fare and the venue's terms."""
params = {{ members = "list", fares = "list", venue = "list", days = "integer" }}
function = "offsite_figures"

[scenarios.offsite]
description = """Which venue to hold a team's offsite at. Every branch checks that everyone fits and that the total \\
cost is within budget; the kept branches are ranked by total cost, lowest first."""
params = {{ team = "string", days = "integer" }}
levers = {{ venue = ["V1", "V2", "V3"] }}     # the only values the agent may choose
branch = "offsite_branch"
objective = "total_cost"
maximise = false
limits = ["capacity", "budget"]
'''

NODES = '''\
id,kind,name,headcount,capacity,day_rate,catering_per_head
T1,team,Platform team,,,,
LON,city,London,,,,
MAN,city,Manchester,,,,
EDI,city,Edinburgh,,,,
P_LON,people,Platform team in London,9,,,
P_MAN,people,Platform team in Manchester,6,,,
P_EDI,people,Platform team in Edinburgh,3,,,
V1,venue,Thames Loft,,20,2400,45
V2,venue,Northern Works,,16,1500,35
V3,venue,Leith Studio,,24,1800,40
'''

EDGES = '''\
source,relation,target,fare
P_LON,member_of,T1,
P_MAN,member_of,T1,
P_EDI,member_of,T1,
P_LON,based_in,LON,
P_MAN,based_in,MAN,
P_EDI,based_in,EDI,
V1,located_in,LON,
V2,located_in,MAN,
V3,located_in,EDI,
LON,travel_to,LON,0
LON,travel_to,MAN,90
LON,travel_to,EDI,140
MAN,travel_to,LON,90
MAN,travel_to,MAN,0
MAN,travel_to,EDI,110
EDI,travel_to,LON,140
EDI,travel_to,MAN,110
EDI,travel_to,EDI,0
'''

LIMIT = '''\
<?xml version="1.0" encoding="UTF-8"?>
<!-- A limit as a DMN decision table: its input's typeRef gives the parameter's type, and FIRST takes the first rule
     that matches. docs/environment.md lists the DMN that is supported. -->
<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" id="{name}_definitions" name="{name}"
             namespace="https://example.org/offsite">
  <decision id="{name}" name="{name}">
    <decisionTable id="{name}_table" hitPolicy="FIRST">
      <input id="{name}_in1" label="{input}"><inputExpression typeRef="number"><text>{input}</text></inputExpression></input>
      <output id="{name}_out1" name="allowed" typeRef="boolean"/>
      <output id="{name}_out2" name="reason" typeRef="string"/>
      <rule id="{name}_refused"><inputEntry><text>{refused}</text></inputEntry><outputEntry><text>false</text></outputEntry><outputEntry><text>"{no}"</text></outputEntry></rule>
      <rule id="{name}_allowed"><inputEntry><text>-</text></inputEntry><outputEntry><text>true</text></outputEntry><outputEntry><text>"{yes}"</text></outputEntry></rule>
    </decisionTable>
  </decision>
</definitions>
'''

FUNCTIONS = '''\
"""The offsite environment's functions: the named queries on the built-in graph, the calculation, and the
scenario's branch.

A query looks up what was asked for and fails when it is not there: an unknown team must be an error, not a team of
nobody whose offsite costs nothing. The branch asks for everything through `steps`, so each query, rule and
calculation is recorded with its numbers; it moves rows between steps but does no arithmetic of its own.
"""


def _known(graph, node_id, kind):
    """The node, if it exists and is of that kind."""
    try:
        node = graph.node(node_id)
    except KeyError:
        node = None
    if node is None or node["kind"] != kind:
        raise LookupError(f"no {{kind}} {{node_id}}")
    return node


def team_members(graph, team):
    """One row per city the team's people are based in."""
    _known(graph, team, "team")
    rows = []
    for edge in graph.edges(relation="member_of", target=team):
        group = graph.node(edge["source"])
        city = graph.neighbours(group["id"], "based_in")[0]
        rows.append({{"city": city["id"], "city_name": city["name"], "headcount": group["headcount"]}})
    return rows


def venue_facts(graph, venue):
    """The venue's terms and city."""
    node = _known(graph, venue, "venue")
    city = graph.neighbours(venue, "located_in")[0]
    return [{{"id": venue, "name": node["name"], "city": city["id"], "city_name": city["name"],
             "capacity": node["capacity"], "day_rate": node["day_rate"],
             "catering_per_head": node["catering_per_head"]}}]


def return_fare(graph, origin, destination):
    """The return fare per person between two cities."""
    _known(graph, origin, "city")
    _known(graph, destination, "city")
    return [{{"origin": origin, "destination": destination, "fare": edge["fare"]}}
            for edge in graph.edges(origin, "travel_to", destination)]


def offsite_figures(members, fares, venue, days):
    """Travel, venue, catering and total cost, headcount and spare seats."""
    terms = venue[0]
    fare = {{row["origin"]: row["fare"] for row in fares}}
    headcount = sum(row["headcount"] for row in members)
    travel = sum(row["headcount"] * fare[row["city"]] for row in members)
    venue_cost = terms["day_rate"] * days
    catering = terms["catering_per_head"] * headcount * days
    return {{"headcount": headcount, "spare_seats": terms["capacity"] - headcount, "travel_cost": travel,
            "venue_cost": venue_cost, "catering_cost": catering, "total_cost": travel + venue_cost + catering}}


def offsite_branch(steps, params, levers):
    """One venue: its terms, everyone's fare there, the figures, and both limits."""
    venue = steps.query("venue_facts", venue=levers["venue"])
    members = steps.query("team_members", team=params["team"])
    fares = [row for member in members
             for row in steps.query("return_fare", origin=member["city"], destination=venue[0]["city"])]
    figures = steps.calculate("offsite_figures", members=members, fares=fares, venue=venue, days=params["days"])
    steps.decide("capacity", spare_seats=figures["spare_seats"])
    steps.decide("budget", total_cost=figures["total_cost"])
    return figures
'''


def write(folder: Path) -> Path:
    """Write the starter files into `folder`; -> the environment file."""
    module = re.sub(r"\W", "_", folder.name.lower()).strip("_") + "_functions"
    if module[0].isdigit():
        module = "e_" + module
    for sub in ("graph", "rules"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    environment = folder / "environment.toml"
    environment.write_text(ENVIRONMENT.format(module=module), encoding="utf-8")
    (folder / "graph" / "nodes.csv").write_text(NODES, encoding="utf-8")
    (folder / "graph" / "edges.csv").write_text(EDGES, encoding="utf-8")
    (folder / "rules" / "capacity.dmn").write_text(LIMIT.format(
        name="capacity", input="spare_seats", refused="&lt; 0", no="The venue is too small for everyone",
        yes="Everyone fits"), encoding="utf-8")
    (folder / "rules" / "budget.dmn").write_text(LIMIT.format(
        name="budget", input="total_cost", refused="&gt; 9000", no="Over the offsite budget of 9,000 pounds",
        yes="Within the offsite budget"), encoding="utf-8")
    (folder / f"{module}.py").write_text(FUNCTIONS.format(), encoding="utf-8")
    return environment


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Start a new environment in a folder of its own.")
    parser.add_argument("folder", type=Path, help="the folder to create; it must be new or empty")
    args = parser.parse_args(argv)
    if args.folder.exists() and not args.folder.is_dir():
        print(f"error: {args.folder} is a file; choose a new folder", file=sys.stderr)
        return 2
    if args.folder.exists() and any(args.folder.iterdir()):
        print(f"error: {args.folder} is not empty; choose a new folder", file=sys.stderr)
        return 2
    environment = write(args.folder)
    print(f"Wrote {environment}, its graph, rules and functions. Next:\n"
          f"  uv run clm-call {environment}                       # its tools\n"
          f"  uv run clm-call {environment} scenario_offsite '{{\"team\": \"T1\", \"days\": 2}}'\n"
          f"  uv run clm-server {environment}                     # serve it over MCP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
