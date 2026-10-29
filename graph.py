"""The built-in graph: nodes and edges from two CSV files, held in memory, with the operations named queries use.

    nodes.csv    id,kind,name and any other columns, which become the node's properties
    edges.csv    source,relation,target and any other columns, which become the edge's properties

An empty cell is no property; a cell that reads as a number becomes one.
"""
from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

NODE_COLUMNS = ("id", "kind", "name")
EDGE_COLUMNS = ("source", "relation", "target")


def _value(cell: str):
    for kind in (int, float):
        try:
            return kind(cell)
        except ValueError:
            pass
    return cell


def _rows(path: Path, needed: tuple[str, ...]) -> list[tuple[int, dict]]:
    """-> (line number, row) for each row, without empty cells; raises ValueError naming the file."""
    with open(path, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = [column for column in needed if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: needs the columns {', '.join(needed)}; missing {', '.join(missing)}")
        rows = []
        for line, row in enumerate(reader, start=2):
            row = {key: cell.strip() for key, cell in row.items() if key and cell and cell.strip()}
            empty = [column for column in needed if column not in row]
            if empty:
                raise ValueError(f"{path.name} line {line}: no {empty[0]}")
            rows.append((line, {key: cell if key in needed else _value(cell) for key, cell in row.items()}))
        return rows


class Graph:
    """Nodes by id, and edges, with the walks that named queries are built from."""

    def __init__(self, nodes: dict[str, dict], edges: list[dict]):
        self._nodes = nodes
        self._edges = edges

    @classmethod
    def from_csv(cls, nodes: Path, edges: Path) -> Graph:
        """Load the two files; raises ValueError naming the file and line of the first problem."""
        nodes, edges = Path(nodes), Path(edges)
        by_id: dict[str, dict] = {}
        for line, row in _rows(nodes, NODE_COLUMNS):
            if row["id"] in by_id:
                raise ValueError(f"{nodes.name} line {line}: node '{row['id']}' appears twice")
            by_id[row["id"]] = row
        links = []
        for line, row in _rows(edges, EDGE_COLUMNS):
            for end in ("source", "target"):
                if row[end] not in by_id:
                    raise ValueError(f"{edges.name} line {line}: unknown node '{row[end]}'")
            links.append(row)
        return cls(by_id, links)

    def node(self, node_id: str) -> dict:
        """One node's properties; raises KeyError."""
        return dict(self._nodes[node_id])

    def nodes(self, kind: str | None = None) -> list[dict]:
        """Every node, or every node of one kind."""
        return [dict(node) for node in self._nodes.values() if kind is None or node["kind"] == kind]

    def edges(self, source: str | None = None, relation: str | None = None, target: str | None = None) -> list[dict]:
        """The edges matching whichever of source, relation and target are given."""
        return [dict(edge) for edge in self._edges
                if (source is None or edge["source"] == source) and (relation is None or edge["relation"] == relation)
                and (target is None or edge["target"] == target)]

    def neighbours(self, node_id: str, relation: str | None = None, direction: str = "out") -> list[dict]:
        """The nodes one edge away: "out" follows edges from the node, "in" edges to it, "both" either."""
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be out, in or both")
        found: list[str] = []
        if direction in ("out", "both"):
            found += [edge["target"] for edge in self.edges(source=node_id, relation=relation)]
        if direction in ("in", "both"):
            found += [edge["source"] for edge in self.edges(target=node_id, relation=relation)]
        return [self.node(other) for other in dict.fromkeys(found)]

    def _paths(self, node_id: str, relations: list[str], depth: int) -> dict[str, list[str]]:
        """Node id -> the shortest path to it from node_id, following relations outwards, up to depth steps."""
        paths = {node_id: [node_id]}
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            if len(paths[current]) > depth:
                continue
            for edge in self._edges:
                if edge["source"] == current and edge["relation"] in relations and edge["target"] not in paths:
                    paths[edge["target"]] = paths[current] + [edge["target"]]
                    queue.append(edge["target"])
        del paths[node_id]
        return paths

    def reachable(self, node_id: str, relations: list[str], depth: int) -> list[dict]:
        """The nodes reached from node_id following relations outwards, up to depth steps, each with its "path"."""
        return [{**self.node(other), "path": path} for other, path in self._paths(node_id, relations, depth).items()]

    def single_points_of_failure(self, ids: list[str], relations: list[str], depth: int) -> list[dict]:
        """The nodes every one of ids reaches, each with "via": {id: path}. Losing one would cut off all of them."""
        if not ids:
            return []
        reached = {node_id: self._paths(node_id, relations, depth) for node_id in ids}
        shared = set.intersection(*(set(paths) for paths in reached.values()))
        return [{**self.node(other), "via": {node_id: reached[node_id][other] for node_id in ids}}
                for other in sorted(shared)]
