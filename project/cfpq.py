from pyformlang.cfg import CFG, Terminal, Variable, Epsilon
from pyformlang.rsa import RecursiveAutomaton
import networkx as nx

from project.adjacency_matrix import AdjacencyMatrixFA, intersect_automata
from project.cfg_utils import cfg_to_weak_normal_form
from scipy.sparse import csc_matrix
from pyformlang.finite_automaton import Symbol, State
from project.fa_utils import graph_to_nfa
from project.rsm_utils import boolean_decompress_rsm, get_rsm_st_edges, RsmSt
from project.graph_utils import get_graph_node_edges


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


class GssV:
    rsm_st: RsmSt
    graph_st: int

    def __init__(self, rsm_st: RsmSt, graph_st: int):
        self.rsm_st = rsm_st
        self.graph_st = graph_st

    def __eq__(self, other):
        return (
            isinstance(other, GssV)
            and self.rsm_st == other.rsm_st
            and self.graph_st == other.graph_st
        )

    def __hash__(self):
        return hash((self.rsm_st, self.graph_st))


class Config:
    rsm_st: RsmSt
    graph_st: int
    gss_v: GssV

    def __init__(self, rsm_st: RsmSt, graph_st: int, gss_v: GssV):
        self.rsm_st = rsm_st
        self.graph_st = graph_st
        self.gss_v = gss_v

    def __eq__(self, other):
        return (
            isinstance(other, Config)
            and self.rsm_st == other.rsm_st
            and self.graph_st == other.graph_st
            and self.gss_v == other.gss_v
        )

    def __hash__(self):
        return hash((self.rsm_st, self.graph_st, self.gss_v))


def get_new_configs(
    conf: Config,
    gss: nx.MultiDiGraph,
    graph: nx.DiGraph,
    rsm: RecursiveAutomaton,
    res: set[tuple[int, int]],
    init_gss_v: GssV,
) -> set[Config]:
    new_configs = set()
    graph_edges: dict[any, set[any]] = get_graph_node_edges(
        nx.MultiDiGraph(graph), conf.graph_st
    )
    rsm_edges: dict[Symbol, set[RsmSt]] = get_rsm_st_edges(rsm, conf.rsm_st)

    # 1.
    labels = set(graph_edges.keys()) & set(rsm_edges.keys())
    for lbl in labels:
        for rsm_st in rsm_edges[lbl]:
            for graph_st in graph_edges[lbl]:
                new_configs.add(Config(rsm_st, graph_st, conf.gss_v))

    # 2.
    for rsm_lbl in rsm_edges.keys():
        if rsm_lbl in rsm.labels:
            for rsm_start_st in rsm.get_box(rsm_lbl).start_state:
                new_rsm_st = RsmSt(rsm_lbl, rsm_start_st)
                new_gss_v = GssV(new_rsm_st, conf.graph_st)

                if new_gss_v in gss.nodes and gss.nodes[new_gss_v]["pop_set"]:
                    for graph_st in gss.nodes[new_gss_v]["pop_set"]:
                        for rsm_st in rsm_edges[rsm_lbl]:
                            gss.add_edge(new_gss_v, conf.gss_v, label=rsm_st)
                            new_configs.add(Config(rsm_st, graph_st, conf.gss_v))
                    continue

                for rsm_st in rsm_edges[rsm_lbl]:
                    gss.add_node(new_gss_v, pop_set=None)
                    gss.add_edge(new_gss_v, conf.gss_v, label=rsm_st)

                new_configs.add(Config(new_rsm_st, conf.graph_st, new_gss_v))

    # 3.
    if conf.rsm_st.st in rsm.get_box(conf.rsm_st.nonterm).final_states:
        if gss.nodes[conf.gss_v]["pop_set"] is None:
            gss.nodes[conf.gss_v]["pop_set"] = set()
        gss.nodes[conf.gss_v]["pop_set"].add(conf.graph_st)

        gss_edges: dict[RsmSt, set[GssV]] = get_graph_node_edges(gss, conf.gss_v)
        for lbl in gss_edges.keys():
            for gss_v in gss_edges[lbl]:
                if gss_v == init_gss_v:
                    res.add((conf.gss_v.graph_st, conf.graph_st))
                    continue
                new_configs.add(Config(lbl, conf.graph_st, gss_v))

    return new_configs


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    states = set(n for n in graph.nodes)
    start_nodes = set(n for n in start_nodes) if start_nodes else states
    final_nodes = set(n for n in final_nodes) if final_nodes else states

    queue: set[Config] = set()
    processed_config: set[Config] = set()
    gss = nx.MultiDiGraph()
    init_gss_v = GssV(RsmSt(Symbol("$"), State(0)), -1)
    res = set()

    # Initialization of configurations and gss nodes
    for rsm_start in rsm.get_box(rsm.initial_label).start_state:
        for graph_st in start_nodes:
            rsm_st = RsmSt(rsm.initial_label, rsm_start)
            gss_v = GssV(rsm_st, graph_st)

            gss.add_node(gss_v, pop_set=None)
            gss.add_edge(gss_v, init_gss_v, label=rsm_st)
            config = Config(rsm_st, graph_st, gss_v)
            queue.add(config)

    # Configuration processing
    while queue:
        config = queue.pop()
        if config in processed_config:
            continue

        processed_config.add(config)
        queue |= get_new_configs(config, gss, graph, rsm, res, init_gss_v)

    return {
        (start_st, final_st)
        for start_st, final_st in res
        if start_st in start_nodes and final_st in final_nodes
    }
