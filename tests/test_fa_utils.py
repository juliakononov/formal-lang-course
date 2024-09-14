import pytest
import networkx as nx

from project.fa_utils import regex_to_dfa
from project.fa_utils import graph_to_nfa
from project.graph_utils import read_graph_from_dot
from project.graph_utils import get_graph
from project.graph_utils import create_and_save_two_cycles_graph
from pyformlang.finite_automaton import State
from pyformlang.finite_automaton import Symbol


class TestRegexToDfa:
    def test_regex_to_dfa_empty(self):
        dfa = regex_to_dfa("")
        assert dfa.is_deterministic()
        assert dfa.is_empty()

    def test_regex_to_dfa(self):
        dfa = regex_to_dfa("abc|d")
        assert dfa.accepts([Symbol("abc")])
        assert dfa.accepts([Symbol("d")])
        assert not dfa.accepts([Symbol("a"), Symbol("b"), Symbol("c")])


class TestGraphToNfa:
    def test_two_cycles_graph_to_nfa(self, tmp_path):
        l1, l2, n1, n2 = "a", "b", 9, 15

        path = tmp_path / "tmp_file.dot"
        create_and_save_two_cycles_graph(n1, n2, (l1, l2), path)
        graph = nx.MultiDiGraph(read_graph_from_dot(path))
        nfa = graph_to_nfa(graph, set(), set())

        assert (
            nfa.states
            == nfa.start_states
            == nfa.final_states
            == set(State(n) for n in graph.nodes)
        )
        assert nfa.symbols == {l1, l2}

    def test_two_cycles_graph_to_nfa_with_start_and_final(self, tmp_path):
        l1, l2, n1, n2 = "a", "b", 20, 15
        start_st = {1, 2, 3}
        final_st = {4, 5, 6, 7, 8, 9}

        path = tmp_path / "tmp_file.dot"
        create_and_save_two_cycles_graph(n1, n2, (l1, l2), path)
        graph = nx.MultiDiGraph(read_graph_from_dot(path))
        nfa = graph_to_nfa(graph, start_st, final_st)

        assert len(nfa.start_states) == len(start_st)
        assert len(nfa.final_states) == len(final_st)
        assert len(set(int(st.value) for st in nfa.states)) == n1 + n2 + 1

    @pytest.mark.parametrize(
        "graph_name,start_st,final_st",
        [
            pytest.param("wc", {1}, {2}, id="wc_with__start_and_final"),
            pytest.param("wc", set(), set(), id="wc"),
        ],
    )
    def test_graph_to_nfa_from_dataset(
        self, graph_name: str, start_st: set[int], final_st: set[int]
    ):
        graph = get_graph(graph_name)
        nfa = graph_to_nfa(graph, start_st, final_st)

        if len(start_st) == len(final_st) == 0:
            assert nfa.start_states == nfa.final_states == nfa.states
        else:
            assert len(nfa.start_states) == len(start_st)
            assert len(nfa.final_states) == len(final_st)
            assert len(set(int(st.value) for st in nfa.states)) == len(graph.nodes)
