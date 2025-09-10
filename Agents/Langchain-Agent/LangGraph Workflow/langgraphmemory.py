from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage
import os
from dotenv import load_dotenv

load_dotenv()
Groq_api_key = os.getenv("GROQ_API_KEY")
Tavily_api_key = os.getenv("TAVILY_API_KEY")

class State(TypedDict):
    messages: Annotated[list, add_messages]

workflow = StateGraph(State)

llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", ToolNode(tools=[tool]))

workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")
workflow.add_edge(START, "chatbot")

# Memory
memory = InMemorySaver()
graph = workflow.compile(checkpointer=memory)

# One thread id = one long-lived memory stream/session
config = {"configurable": {"thread_id": "1"}}

# Track which AI messages we've already printed (avoid duplicates during streaming)
_seen_msg_ids = set()

def stream_graph_updates(user_input: str):
    for state in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    ):
        # Print only fresh, normal AI messages (skip tool-calls)
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                mid = getattr(msg, "id", None) or id(msg)
                if mid not in _seen_msg_ids:
                    print("Assistant:", msg.content)
                    _seen_msg_ids.add(mid)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about Agents in Generative AI?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break