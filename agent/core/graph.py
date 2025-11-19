from langgraph.graph import END, StateGraph

from agent.core.nodes import chat_node, retrieval_node, guardrail_node
from agent.core.state import State
from agent.core.telemetry import langfuse_handler


def build_graph():
    graph = StateGraph(State)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("chat", chat_node)

    graph.set_entry_point("guardrail")

    graph.add_conditional_edges(
        "guardrail",
        lambda state: "end" if state["guardrail_message"] is not None else "continue",
        {"end": END, "continue": "retrieval"},
    )

    graph.add_edge("retrieval", "chat")
    graph.add_edge("chat", END)

    return graph.compile().with_config({"callbacks": [langfuse_handler]})
