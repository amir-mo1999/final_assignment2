# src/core/nodes.py
from langchain_core.messages import AIMessage
from .state import State
from .llm import llm

def chat_node(state: State) -> State:
    """Single node that takes the conversation so far and returns the next reply."""
    response = llm.invoke(state["messages"])
    state["messages"].append(response if isinstance(response, AIMessage) else AIMessage(content=str(response)))
    return state
