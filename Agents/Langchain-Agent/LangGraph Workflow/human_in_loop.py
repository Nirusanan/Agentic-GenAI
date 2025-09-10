from typing import Annotated
from typing_extensions import TypedDict
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
import os
from dotenv import load_dotenv

load_dotenv()
Groq_api_key = os.getenv("GROQ_API_KEY")
Tavily_api_key = os.getenv("TAVILY_API_KEY")

class State(TypedDict):
    messages: Annotated[list, add_messages]

workflow = StateGraph(State)

# Human assistance tool
@tool
def human_assistance(query: str) -> str:
    """Request clarification or missing information from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)

# Chatbot node
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}

workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode(tools=tools))

workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")
workflow.add_edge("chatbot", END)
workflow.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = workflow.compile(checkpointer=memory)

# ---------------- Chat Loop ----------------
_seen_msg_ids = set()
config = {"configurable": {"thread_id": "2"}}


def stream_graph_updates(user_input: str):
    for state in graph.stream(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI. If the user request is clear and can be answered "
                        "directly or using available tools (like TavilySearch), do so. "
                        "If important information is missing and you cannot proceed, "
                        "then call the `human_assistance` tool to ask the user for clarification."
                    ),
                },
                {"role": "user", "content": user_input},
            ]
        },
        config,
        stream_mode="values",
    ):
        for msg in state["messages"]:
            # 1. Print final assistant replies (no tool calls)
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                mid = getattr(msg, "id", None) or id(msg)
                if mid not in _seen_msg_ids:
                    print("Assistant:", msg.content)
                    _seen_msg_ids.add(mid)

            # 2. Print tool results
            elif isinstance(msg, ToolMessage):
                tid = getattr(msg, "id", None) or id(msg)
                if tid not in _seen_msg_ids:
                    print("Tool Result:", msg.content)
                    _seen_msg_ids.add(tid)

            # 3. Handle clarification requests
            elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    if (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")) == "human_assistance":
                        print("\n(Assistant needs clarification)")
                        human_reply = input("User (Your Clarification): ")
                        command = Command(resume={"data": human_reply})

                        for resumed in graph.stream(command, config, stream_mode="values"):
                            for rmsg in resumed["messages"]:
                                if isinstance(rmsg, AIMessage) and not getattr(rmsg, "tool_calls", None):
                                    rid = getattr(rmsg, "id", None) or id(rmsg)
                                    if rid not in _seen_msg_ids:
                                        print("Assistant:", rmsg.content)
                                        _seen_msg_ids.add(rid)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
