"""Tests for graph.py: the built-in graph, loaded from a nodes file and an edges file."""
# Test names say what they check, and pytest fixtures are used by repeating their names.
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest

from clm_decision.graph import Graph

NODES = """id,kind,name,defect_rate,note
S1,supplier,North Mill,0.01,
S2,supplier,South Mill,0.02,new
S3,supplier,East Mill,0.015,
R1,material,Resin,,
R2,material,Steel,,
M1,mine,Deep Pit,,
"""

EDGES = """source,relation,target,share
S1,depends_on,R1,1
S2,depends_on,R1,1
S3,depends_on,R2,1
R1,depends_on,M1,
"""


@pytest.fixture
def graph(tmp_path):
    (tmp_path / "nodes.csv").write_text(NODES, encoding="utf-8")
    (tmp_path / "edges.csv").write_text(EDGES, encoding="utf-8")
    return Graph.from_csv(tmp_path / "nodes.csv", tmp_path / "edges.csv")


def ids(nodes):
    return {node["id"] for node in nodes}


def test_properties_are_parsed(graph):
    assert graph.node("S1")["defect_rate"] == 0.01
    assert graph.node("S1")["name"] == "North Mill"
    assert "note" not in graph.node("S1")
    assert graph.node("S2")["note"] == "new"
    assert graph.edges(source="S1")[0]["share"] == 1


def test_only_a_plain_decimal_becomes_a_number(tmp_path):
    cells = ["nan", "inf", "1e5", "1_000", "-3", "2.50", "007", "1.", ".5", "+4"]
    (tmp_path / "nodes.csv").write_text("id,kind,name,cell\n" + "".join(f"N{i},n,n,{cell}\n"
                                                                    for i, cell in enumerate(cells)), encoding="utf-8")
    (tmp_path / "edges.csv").write_text("source,relation,target\n", encoding="utf-8")
    graph = Graph.from_csv(tmp_path / "nodes.csv", tmp_path / "edges.csv")
    assert [graph.node(f"N{i}")["cell"] for i in range(len(cells))] == \
        ["nan", "inf", "1e5", "1_000", -3, 2.5, 7, "1.", ".5", "+4"]
    assert isinstance(graph.node("N4")["cell"], int) and isinstance(graph.node("N5")["cell"], float)


def test_neighbours_follow_the_direction(graph):
    assert ids(graph.neighbours("R1", "depends_on", "in")) == {"S1", "S2"}
    assert ids(graph.neighbours("R1", "depends_on", "out")) == {"M1"}
    assert ids(graph.neighbours("R1", direction="both")) == {"S1", "S2", "M1"}


def test_reachable_keeps_the_path(graph):
    reached = {node["id"]: node for node in graph.reachable("S1", ["depends_on"], 2)}
    assert reached["M1"]["path"] == ["S1", "R1", "M1"]
    assert "M1" not in ids(graph.reachable("S1", ["depends_on"], 1))


def test_single_points_of_failure(graph):
    shared = {node["id"]: node for node in graph.single_points_of_failure(["S1", "S2"], ["depends_on"], 3)}
    assert set(shared) == {"R1", "M1"}
    assert shared["R1"]["via"] == {"S1": ["S1", "R1"], "S2": ["S2", "R1"]}
    assert not graph.single_points_of_failure(["S1", "S3"], ["depends_on"], 3)


def test_an_edge_to_an_unknown_node_is_an_error(tmp_path):
    (tmp_path / "nodes.csv").write_text(NODES, encoding="utf-8")
    (tmp_path / "edges.csv").write_text(EDGES + "S1,depends_on,X9,\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"edges.csv line 6: unknown node 'X9'"):
        Graph.from_csv(tmp_path / "nodes.csv", tmp_path / "edges.csv")
