"""Graph nodes for retrieval-augmented generation."""

from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.core.guardrails import (
    FALLBACK_MESSAGE,
    GuardrailViolation,
    ensure_supported_query,
)
from agent.core.llm import get_llm
from agent.core.retrieval import similarity_search
from agent.core.state import State
from agent.core.telemetry import get_telemetry_client

_llm = get_llm()
_telemetry = get_telemetry_client()


def _last_user_message(messages: List) -> HumanMessage | None:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


def retrieval_node(state: State) -> State:
    messages = state.get("messages", [])
    user_message = _last_user_message(messages)
    if not user_message:
        return {"retrieved_context": [], "guardrail_message": FALLBACK_MESSAGE}

    try:
        ensure_supported_query(str(user_message.content))
    except GuardrailViolation as exc:
        return {"retrieved_context": [], "guardrail_message": str(exc)}

    result = similarity_search(str(user_message.content))
    if result.error:
        return {"retrieved_context": [], "guardrail_message": result.error}

    return {"retrieved_context": result.chunks, "guardrail_message": None}


def _format_context(chunks: List[dict]) -> str:
    if not chunks:
        return "No matching code chunks were retrieved."

    formatted = []
    for chunk in chunks:
        header = (
            f"File: {chunk['file_path']} "
            f"(chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']})"
        )
        formatted.append(f"{header}\n{chunk['content']}")
    return "\n\n".join(formatted)


def chat_node(state: State) -> State:
    messages = state.get("messages", [])
    context = state.get("retrieved_context", [])
    guardrail_message = state.get("guardrail_message")
    user_message = _last_user_message(messages)

    if guardrail_message:
        response_text = guardrail_message
    elif not user_message:
        response_text = FALLBACK_MESSAGE
    else:
        system_prompt = (
            "You are a meticulous assistant that answers questions about an ingested "
            "Python codebase. Only use the retrieved code context. If the context "
            "does not contain the answer, reply with 'I do not know based on the "
            "available repository context.' Reference file paths in your answer."
        )
        context_text = _format_context(context)
        prompt_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"User question: {user_message.content}\n\n"
                    f"Retrieved context:\n{context_text}\n\n"
                    "Answer the question using only this information."
                )
            ),
        ]
        response = _llm.invoke(prompt_messages)
        response_text = (
            str(response.content) if isinstance(response, AIMessage) else str(response)
        )

    _telemetry.log_interaction(
        user_query=str(user_message.content) if user_message else "",
        response=response_text,
        retrieved_context=context,
        error=guardrail_message,
    )

    return {
        "messages": [AIMessage(content=response_text)],
        "retrieved_context": context,
        "guardrail_message": None,
    }
