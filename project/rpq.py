from networkx import MultiDiGraph
from project.fa_utils import regex_to_dfa, graph_to_nfa
from project.adjacency_matrix import AdjacencyMatrixFA, intersect_automata


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    reg_adj_matrix = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_adj_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    intersect = intersect_automata(reg_adj_matrix, graph_adj_matrix)
    res_matrix = intersect.transitive_closure()

    valid_pairs = set()

    for start_st in intersect.start_sts:
        for final_st in intersect.final_sts:
            if res_matrix[intersect.st_to_idx[start_st], intersect.st_to_idx[final_st]]:
                start = start_st.value[1].value
                final = final_st.value[1].value

                valid_pairs.add((start, final))

    return valid_pairs
