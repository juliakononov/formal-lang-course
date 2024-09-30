import itertools
from typing import Iterable
from pyformlang.finite_automaton import Symbol, State
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from scipy.sparse import csc_matrix, kron


class AdjacencyMatrixFA:
    def __init__(self, fa: NondeterministicFiniteAutomaton | None):
        if fa is None:
            self.num_sts = 0
            self.st_to_idx = {}
            self.idx_to_st = {}
            self.start_sts = set()
            self.final_sts = set()
            self.adjacency_matrices = {}
            return

        self.num_sts = len(fa.states)
        self.st_to_idx: dict[State, int] = {st: i for i, st in enumerate(fa.states)}
        self.idx_to_st: dict[int, State] = {i: st for st, i in self.st_to_idx.items()}
        self.start_sts: set[State] = fa.start_states
        self.final_sts: set[State] = fa.final_states
        self.adjacency_matrices: dict[Symbol, csc_matrix] = {}

        for from_st in self.st_to_idx.keys():
            from_idx = self.st_to_idx[from_st]
            transitions: dict[Symbol, State | set[State]] = fa.to_dict().get(from_st)
            if transitions is None:
                continue
            for symbol in transitions.keys():
                if symbol not in self.adjacency_matrices:
                    self.adjacency_matrices[symbol] = csc_matrix(
                        (self.num_sts, self.num_sts), dtype=bool
                    )
                if isinstance(transitions[symbol], Iterable):
                    for to_st in transitions[symbol]:
                        to_idx = self.st_to_idx[to_st]
                        self.adjacency_matrices[symbol][from_idx, to_idx] = True
                else:
                    to_st: State = transitions[symbol]
                    to_idx = self.st_to_idx[to_st]
                    self.adjacency_matrices[symbol][from_idx, to_idx] = True
        return

    def accepts(self, word: Iterable[Symbol]) -> bool:
        cur_configs: set[State] = self.start_sts

        for symbol in word:
            if symbol not in self.adjacency_matrices:
                return False

            next_configs = set()
            transition_matrix = self.adjacency_matrices[symbol]

            for cur_st in cur_configs:
                for st in self.st_to_idx.keys():
                    if transition_matrix[self.st_to_idx[cur_st], self.st_to_idx[st]]:
                        next_configs.add(st)

            cur_configs = next_configs

        return any(final_st in cur_configs for final_st in self.final_sts)

    def transitive_closure(self) -> csc_matrix:
        res = csc_matrix((self.num_sts, self.num_sts), dtype=bool)
        res.setdiag(True)

        for matrix in self.adjacency_matrices.values():
            res += matrix

        res.astype(bool)
        changed = True

        while changed:
            new_res = (res @ res).astype(bool)

            if (new_res != res).nnz == 0:
                changed = False
            else:
                res = new_res
        return res

    def is_empty(self) -> bool:
        transitive_closure = self.transitive_closure()

        if transitive_closure is None:
            return True

        for start_st, final_st in itertools.product(self.start_sts, self.final_sts):
            if transitive_closure[self.st_to_idx[start_st], self.st_to_idx[final_st]]:
                return False
        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    res = AdjacencyMatrixFA(None)
    M1, M2 = automaton1.adjacency_matrices, automaton2.adjacency_matrices

    res.num_sts = automaton1.num_sts * automaton2.num_sts
    symbols = set(M1.keys()) & set(M2.keys())
    res.adjacency_matrices = {s: kron(M1[s], M2[s], format="csc") for s in symbols}

    for st1, st2 in itertools.product(
        automaton1.st_to_idx.keys(), automaton2.st_to_idx.keys()
    ):
        idx1, idx2 = automaton1.st_to_idx[st1], automaton2.st_to_idx[st2]
        res_idx = idx1 * automaton2.num_sts + idx2

        res.st_to_idx[State((st1, st2))] = res_idx
        if st1 in automaton1.start_sts and st2 in automaton2.start_sts:
            res.start_sts.add(State((st1, st2)))
        if st1 in automaton1.final_sts and st2 in automaton2.final_sts:
            res.final_sts.add(State((st1, st2)))
    return res
