from langgraph.graph import END, StateGraph

from agent.core.nodes import chat_node, retrieval_node
from agent.core.state import State


def build_graph():
    graph = StateGraph(State)

    graph.add_node("retrieval", retrieval_node)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("retrieval")

    graph.add_edge("retrieval", "chat")
    graph.add_edge("chat", END)

    return graph.compile()
