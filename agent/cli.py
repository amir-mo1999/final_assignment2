# src/cli.py
from langchain_core.messages import HumanMessage
from core.graph import build_graph

def run_cli():
    graph = build_graph()
    messages = []

    print("Simple LangGraph agent. Type 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append(HumanMessage(content=user_input))

        state = {"messages": messages}
        new_state = graph.invoke(state)

        # last message should be the assistant reply
        assistant_msg = new_state["messages"][-1]
        print(f"Agent: {assistant_msg.content}")

        messages = new_state["messages"]
