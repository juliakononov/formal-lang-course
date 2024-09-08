from project.graph_utils import *
from pathlib import Path

import pytest
import networkx as nx

from scripts.shared import TESTS


def test_get_graph_data_from_bzip_graph():
    graph_data = get_graph_data("wc")
    expected_graph_data = GraphData(num_nodes=332, num_edges=269, labels={"d", "a"})
    assert graph_data == expected_graph_data


def test_get_graph_data_from_enzyme_graph():
    graph_data = get_graph_data("enzyme")
    expected_graph_data = GraphData(
        num_nodes=48815,
        num_edges=86543,
        labels={
            "activity",
            "altLabel",
            "broaderTransitive",
            "cofactorLabel",
            "comment",
            "imports",
            "label",
            "narrowerTransitive",
            "obsolete",
            "prefLabel",
            "replacedBy",
            "replaces",
            "subClassOf",
            "type",
        },
    )
    assert graph_data == expected_graph_data


def test_get_graph_data_fail():
    with pytest.raises(FileNotFoundError):
        get_graph_data("unknown_graph_name")


def test_create_and_save_two_cycles_graph(tmp_path):
    expected_graph = read_graph_from_dot(
        TESTS / Path("resources/test_graph_utils/test_create_and_save_two_cycles_graph")
    )

    path = tmp_path / "tmp_file.dot"
    create_and_save_two_cycles_graph(3, 4, ("a", "b"), path)
    graph = read_graph_from_dot(path)

    assert nx.is_isomorphic(
        graph, expected_graph, edge_match=dict.__eq__, node_match=dict.__eq__
    )


def test_create_and_save_two_cycles_graph_fail(tmp_path):
    path = tmp_path / "tmp_file.dot"
    with pytest.raises(nx.NetworkXError):
        create_and_save_two_cycles_graph(0, -3, ("a", "b"), path)
