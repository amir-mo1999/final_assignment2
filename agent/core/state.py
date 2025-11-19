from typing import Annotated, List, TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved_context: List[dict]
    guardrail_message: str | None
