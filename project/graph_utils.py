from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Set

import cfpq_data
import networkx as nx

__all__ = [
    "GraphData",
    "get_graph_data",
    "read_graph_from_dot",
    "get_graph",
    "create_and_save_two_cycles_graph",
    "get_graph_node_edges",
]


@dataclass
class GraphData:
    num_nodes: int
    num_edges: int
    labels: Set[Any]


def get_graph(graph_name: str) -> nx.MultiDiGraph:
    path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(path)
    return graph


def get_graph_data(graph_name: str) -> GraphData:
    graph = get_graph(graph_name)

    return GraphData(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        set(cfpq_data.get_sorted_labels(graph)),
    )


def save_graph_to_dot(graph: nx.MultiDiGraph, output_file: Path) -> None:
    nx.drawing.nx_pydot.write_dot(graph, output_file)


def read_graph_from_dot(path: Path) -> nx.DiGraph:
    return nx.DiGraph(nx.drawing.nx_pydot.read_dot(path))


def get_graph_node_edges(g: nx.MultiDiGraph, from_nd: Any) -> dict[Any, set[Any]]:
    edges = {}
    for _, to_nd, lbl in g.edges(from_nd, data="label"):
        edges.setdefault(lbl, set()).add(to_nd)
    return edges


def create_and_save_two_cycles_graph(
    num_first_cycle_nodes: int,
    num_second_cycle_nodes: int,
    labels: Tuple[str, str],
    output_file: Path,
) -> None:
    graph = cfpq_data.labeled_two_cycles_graph(
        num_first_cycle_nodes, num_second_cycle_nodes, labels=labels
    )
    save_graph_to_dot(graph, output_file)
