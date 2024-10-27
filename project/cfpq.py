from pyformlang.cfg import CFG, Terminal, Variable, Epsilon
import networkx as nx
from project.cfg_utils import cfg_to_weak_normal_form


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
