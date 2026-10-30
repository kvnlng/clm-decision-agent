"""Copy an environment's CSV graph into Neo4j, replacing what the database holds.

    set -a; . ./.neo4j-local.env; set +a
    uv run clm-load-graph ENVIRONMENT

Each node becomes (:Node:<Kind> {id, kind, name, ...}), its kind capitalised as a label, and each edge a relationship
whose type is the relation in capitals, with the edge's other columns as its properties. The connection comes from
the environment's [graph.neo4j] settings and the password from the variable they name. Then set
`[graph] backend = "neo4j"`, and the named queries run their cypher on the database.
"""
from __future__ import annotations

import argparse
import re
import sys
from itertools import groupby

from .environment import graph_settings, load_graph_files, neo4j_password
from .graph import EDGE_COLUMNS, Graph
from .scenario import ScenarioError

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
BATCH = 1000


def _label(text: str, what: str) -> str:
    """A kind or relation as a Neo4j label or type; they cannot be parameters, so only plain names are allowed."""
    if not _NAME.fullmatch(text):
        raise ValueError(f"{what} '{text}' cannot be a Neo4j label: use letters, digits and _")
    return text


def load(graph: Graph, driver, database: str) -> dict:
    """Replace the database's contents with the graph -> {"nodes": n, "edges": n}."""
    nodes = sorted(graph.nodes(), key=lambda node: node["kind"])
    edges = sorted(graph.edges(), key=lambda edge: edge["relation"])
    statements = [("MATCH (n) DETACH DELETE n", {}),
                  ("CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE", {})]
    for kind, group in groupby(nodes, key=lambda node: node["kind"]):
        label = _label(kind, "kind")
        label = label[:1].upper() + label[1:]
        rows = list(group)
        for start in range(0, len(rows), BATCH):
            statements.append((f"UNWIND $rows AS row CREATE (n:Node:{label}) SET n = row",
                               {"rows": rows[start:start + BATCH]}))
    for relation, group in groupby(edges, key=lambda edge: edge["relation"]):
        kind = _label(relation, "relation").upper()
        rows = [{"source": edge["source"], "target": edge["target"],
                 "properties": {key: value for key, value in edge.items() if key not in EDGE_COLUMNS}}
                for edge in group]
        for start in range(0, len(rows), BATCH):
            statements.append((f"UNWIND $rows AS row MATCH (a:Node {{id: row.source}}), (b:Node {{id: row.target}}) "
                               f"CREATE (a)-[r:{kind}]->(b) SET r = row.properties",
                               {"rows": rows[start:start + BATCH]}))
    with driver.session(database=database) as session:
        for cypher, parameters in statements:
            session.run(cypher, parameters)
    return {"nodes": len(nodes), "edges": len(edges)}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Copy an environment's CSV graph into Neo4j.")
    parser.add_argument("environment", help="path to the environment's environment.toml")
    args = parser.parse_args(argv)
    try:
        from .graph_neo4j import connect, driver_errors  # pylint: disable=import-outside-toplevel
        settings = graph_settings(args.environment)
        graph = load_graph_files(settings)
        driver = connect(settings.uri, settings.user, neo4j_password(settings))
    except (ScenarioError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        counts = load(graph, driver, settings.database)
    except (ValueError, *driver_errors()) as error:
        print(f"error: {str(error).strip().splitlines()[0]}", file=sys.stderr)
        return 2
    finally:
        driver.close()
    print(f"Loaded {counts['nodes']} nodes and {counts['edges']} edges into {settings.uri} ({settings.database}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
