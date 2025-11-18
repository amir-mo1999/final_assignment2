"""Console interface for the LangGraph agent."""

from __future__ import annotations

from textwrap import shorten

from langchain_core.messages import HumanMessage

from agent.core.graph import build_graph
from agent.core.state import State


def _format_preview(content: str) -> str:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if not lines:
        return "(empty chunk)"
    return shorten(" ".join(lines), width=120, placeholder="...")


def _print_context(chunks: list[dict]) -> None:
    print("Retrieved context:")
    if not chunks:
        print("  (no relevant context returned)")
        return
    for chunk in chunks:
        preview = _format_preview(chunk["content"])
        location = (
            f"{chunk['file_path']} [{chunk['chunk_index'] + 1}/{chunk['total_chunks']}]"
        )
        print(f"  - {location}: {preview}")


def run_cli() -> None:
    graph = build_graph()
    messages: list = []

    print("Codebase QA agent. Type 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        state: State = {
            "messages": messages,
            "retrieved_context": [],
            "guardrail_message": None,
        }
        new_state = graph.invoke(state)
        messages = new_state["messages"]
        assistant_msg = messages[-1]
        print(f"Agent: {assistant_msg.content}")
        _print_context(new_state.get("retrieved_context", []))
        print()
