from langgraph.graph import StateGraph, END
from .state import State
from .nodes import chat_node


def build_graph():
    graph = StateGraph(State)

    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")

    # After chat_node, we end the graph run (one turn per invocation)
    graph.add_edge("chat", END)

    return graph.compile()
