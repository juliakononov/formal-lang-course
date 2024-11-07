from pyformlang.cfg import CFG, Terminal, Variable, Epsilon
from pyformlang.rsa import RecursiveAutomaton
import networkx as nx

from project.adjacency_matrix import AdjacencyMatrixFA, intersect_automata
from project.cfg_utils import cfg_to_weak_normal_form
from scipy.sparse import csc_matrix
from pyformlang.finite_automaton import Symbol, State
from project.fa_utils import graph_to_nfa
from project.rsm_utils import boolean_decompress_rsm


def classify_productions(productions):
    epsilon_prods: set[Variable] = set()  # (A -> ε)
    term_prods: dict[Terminal, set[Variable]] = {}  # (A -> a)
    nonterm_prods: dict[(Variable, Variable), set[Variable]] = {}  # (A -> BC)

    for production in productions:
        head, body = production.head, production.body

        if len(body) == 0 or isinstance(body[0], Epsilon):
            epsilon_prods.add(head)
        elif len(body) == 1 and isinstance(body[0], Terminal):
            term_prods.setdefault(body[0].value, set()).add(head)
        else:
            nonterm_prods.setdefault((body[0], body[1]), set()).add(head)

    return epsilon_prods, term_prods, nonterm_prods


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    cfg = cfg_to_weak_normal_form(cfg)

    epsilon_prods, term_prods, nonterm_prods = classify_productions(cfg.productions)
    adjacency_matrix = {(n, n, eps) for n in graph.nodes for eps in epsilon_prods} | {
        (start, final, term)
        for start, final, lbl in graph.edges.data("label")
        if lbl in term_prods
        for term in term_prods[lbl]
    }
    queue = adjacency_matrix.copy()

    while queue:
        new_nonterm = set()
        start1, final1, nonterm1 = queue.pop()

        for start2, final2, nonterm2 in adjacency_matrix:
            if final1 == start2 and (nonterm1, nonterm2) in nonterm_prods:
                for nonterm in nonterm_prods[(nonterm1, nonterm2)]:
                    if (start1, final2, nonterm) not in adjacency_matrix:
                        new_nonterm.add((start1, final2, nonterm))
            if final2 == start1 and (nonterm2, nonterm1) in nonterm_prods:
                for nonterm in nonterm_prods[(nonterm2, nonterm1)]:
                    if (start2, final1, nonterm) not in adjacency_matrix:
                        new_nonterm.add((start2, final1, nonterm))

        queue |= new_nonterm
        adjacency_matrix |= new_nonterm

    return {
        (start, final)
        for start, final, nonterm in adjacency_matrix
        if (
            start in start_nodes
            and final in final_nodes
            and nonterm.value == cfg.start_symbol.value
        )
    }


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    cfg = cfg_to_weak_normal_form(cfg)

    epsilon_prods, term_prods, nonterm_prods = classify_productions(cfg.productions)
    node_to_idx = {n: i for i, n in enumerate(graph.nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    num_nodes = graph.number_of_nodes()
    adjacency_matrix: dict[Variable, csc_matrix] = {
        nonterm: csc_matrix((num_nodes, num_nodes), dtype=bool)
        for nonterm in cfg.variables
    }

    for eps in epsilon_prods:
        for n in range(num_nodes):
            adjacency_matrix[eps][n, n] = True

    for s, f, lbl in graph.edges.data("label"):
        if lbl in term_prods:
            for term in term_prods[lbl]:
                adjacency_matrix[term][node_to_idx[s], node_to_idx[f]] = True

    queue = set(cfg.variables)
    while queue:
        updated_var = queue.pop()
        # M_A = M_A + M_B * M_C, if A -> BC in nonterm_prods
        for B, C in nonterm_prods:
            if updated_var != B and updated_var != C:
                continue

            matrix_change = adjacency_matrix[B] @ adjacency_matrix[C]
            for nonterm in nonterm_prods[(B, C)]:
                old_matrix = adjacency_matrix[nonterm]
                adjacency_matrix[nonterm] += matrix_change
                if (old_matrix != adjacency_matrix[nonterm]).count_nonzero() != 0:
                    queue.add(nonterm)

    valid_pairs: set[tuple[int, int]] = set()
    for nonterm in adjacency_matrix:
        if nonterm.value == cfg.start_symbol.value:
            for r, c in zip(*adjacency_matrix[nonterm].nonzero()):
                if idx_to_node[r] in start_nodes and idx_to_node[c] in final_nodes:
                    valid_pairs.add((idx_to_node[r], idx_to_node[c]))

    return valid_pairs


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_m = boolean_decompress_rsm(rsm)
    graph_m = AdjacencyMatrixFA(
        graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes)
    )

    def delta(tc: csc_matrix) -> dict[Symbol, csc_matrix]:
        res: dict[Symbol, csc_matrix] = {}
        for i, j in zip(*tc.nonzero()):
            rsm_i, rsm_j = i % rsm_m.num_sts, j % rsm_m.num_sts
            st1, st2 = rsm_m.idx_to_st[rsm_i], rsm_m.idx_to_st[rsm_j]
            if st1 in rsm_m.start_sts and st2 in rsm_m.final_sts:
                assert st1.value[0] == st2.value[0]
                nonterm = st1.value[0]

                graph_i, graph_j = i // rsm_m.num_sts, j // rsm_m.num_sts
                if (
                    nonterm in graph_m.adjacency_matrices
                    and graph_m.adjacency_matrices[nonterm][graph_i, graph_j]
                ):
                    continue

                if nonterm not in res:
                    res[nonterm] = csc_matrix(
                        (graph_m.num_sts, graph_m.num_sts), dtype=bool
                    )
                res[nonterm][graph_i, graph_j] = True
        return res

    while True:
        transitive_closure = intersect_automata(graph_m, rsm_m).transitive_closure()
        m_delta = delta(transitive_closure)
        if not m_delta:
            break
        for symbol in m_delta.keys():
            if symbol not in graph_m.adjacency_matrices:
                graph_m.adjacency_matrices[symbol] = m_delta[symbol]
            else:
                graph_m.adjacency_matrices[symbol] += m_delta[symbol]

    valid_pairs: set[tuple[int, int]] = set()
    start_m = graph_m.adjacency_matrices.get(rsm.initial_label)
    if start_m is None:
        return valid_pairs

    for start in start_nodes:
        for final in final_nodes:
            if start_m[
                graph_m.st_to_idx[State(start)], graph_m.st_to_idx[State(final)]
            ]:
                valid_pairs.add((start, final))
    return valid_pairs
