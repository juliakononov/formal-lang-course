from typing import Set

from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import NondeterministicTransitionFunction
from pyformlang.finite_automaton import State
from pyformlang.finite_automaton import Symbol
from pyformlang.regular_expression import Regex
from networkx import MultiDiGraph


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    states = set(State(n) for n in graph.nodes)
    start_states = (State(st) for st in start_states) if start_states else states
    final_states = (State(st) for st in final_states) if final_states else states

    nfa = NondeterministicFiniteAutomaton(
        states=states,
        start_state=start_states,
        final_states=final_states,
    )

    for u, v, data in graph.edges(data=True):
        symbol = Symbol(data["label"])
        nfa.add_transition(State(u), symbol, State(v))

    return nfa
