from dataclasses import dataclass

from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from project.adjacency_matrix import AdjacencyMatrixFA
from pyformlang.finite_automaton.epsilon_nfa import EpsilonNFA
from typing import Iterable


@dataclass
class RsmSt:
    nonterm: Symbol
    st: State

    def __eq__(self, other):
        return (
            isinstance(other, RsmSt)
            and self.nonterm == other.nonterm
            and self.st == other.st
        )

    def __hash__(self):
        return hash((self.nonterm, self.st))


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def get_rsm_st_edges(
    rsm: RecursiveAutomaton, from_st: RsmSt
) -> dict[Symbol, set[RsmSt]]:
    nonterm = from_st.nonterm
    edges = {}
    if rsm.get_box(nonterm) is None:
        return {}

    nonterm_edges = rsm.get_box(nonterm).dfa.to_dict()
    if from_st.st in nonterm_edges.keys():
        for lbl, to_st in nonterm_edges[from_st.st].items():
            if not isinstance(to_st, Iterable):
                edges.setdefault(lbl, set()).add(RsmSt(nonterm, to_st))
                continue

            for to_state in to_st:
                edges.setdefault(lbl, set()).add(RsmSt(nonterm, to_state))

    return edges


def rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for nonterm, box in rsm.boxes.items():
        dfa: EpsilonNFA = box.dfa

        for st in dfa.start_states:
            nfa.add_start_state(State((nonterm, st.value)))
        for st in dfa.final_states:
            nfa.add_final_state(State((nonterm, st.value)))

        for from_st in dfa.states:
            transitions: dict[Symbol, State | set[State]] = dfa.to_dict().get(from_st)
            if transitions is None:
                continue
            for symbol in transitions.keys():
                for to_st in (
                    transitions[symbol]
                    if isinstance(transitions[symbol], Iterable)
                    else {transitions[symbol]}
                ):
                    nfa.add_transition(
                        State((nonterm, from_st)), symbol, State((nonterm, to_st))
                    )
    return nfa


def boolean_decompress_rsm(rsm: RecursiveAutomaton) -> AdjacencyMatrixFA:
    return AdjacencyMatrixFA(rsm_to_nfa(rsm))
