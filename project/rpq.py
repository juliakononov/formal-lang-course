from networkx import MultiDiGraph
from project.fa_utils import regex_to_dfa, graph_to_nfa
from project.adjacency_matrix import AdjacencyMatrixFA, intersect_automata
from scipy.sparse import csc_matrix, vstack
from pyformlang.finite_automaton import Symbol


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


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    dfa = AdjacencyMatrixFA(regex_to_dfa(regex))
    nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    nfa_start_idx = list(nfa.st_to_idx[st] for st in nfa.start_sts)
    dfa_start = dfa.st_to_idx[list(dfa.start_sts)[0]]
    front = vstack(
        [
            csc_matrix(
                ([True], ([dfa_start], [nfa_start])),
                (dfa.num_sts, nfa.num_sts),
                dtype=bool,
            )
            for nfa_start in nfa_start_idx
        ]
    )

    visited = front
    symbols = set(dfa.adjacency_matrices.keys()) & set(nfa.adjacency_matrices.keys())
    permutation_matrices: dict[Symbol, csc_matrix] = {
        s: dfa.adjacency_matrices[s].transpose() for s in symbols
    }

    while front.count_nonzero() != 0:
        new_front = front
        for s in symbols:
            front_for_symbol = front @ nfa.adjacency_matrices[s]
            front_after_permutation = vstack(
                [
                    permutation_matrices[s]
                    @ front_for_symbol[(i * dfa.num_sts) : ((i + 1) * dfa.num_sts), :]
                    for i in range(len(start_nodes))
                ]
            )
            new_front += front_after_permutation
        front = new_front.astype(bool) > visited
        visited += front

    valid_pairs = set()
    final_matrix = csc_matrix((dfa.num_sts, nfa.num_sts), dtype=bool)
    for nfa_final in nfa.final_sts:
        for dfa_final in dfa.final_sts:
            final_matrix[dfa.st_to_idx[dfa_final], nfa.st_to_idx[nfa_final]] = True

    _final_st = vstack([final_matrix for _ in range(len(start_nodes))])
    final_st = vstack(
        [
            csc_matrix((dfa.num_sts, nfa.num_sts), dtype=bool)
            for _ in range(len(start_nodes))
        ]
    )
    for i in range(dfa.num_sts * len(start_nodes)):
        for j in range(nfa.num_sts):
            if (_final_st[i, j]) and (visited[i, j]):
                final_st[i, j] = True

    for r, c in zip(*final_st.nonzero()):
        nfa_start = nfa.idx_to_st[nfa_start_idx[(r // dfa.num_sts)]]
        valid_pairs.add((nfa_start.value, nfa.idx_to_st[c].value))
    return valid_pairs
