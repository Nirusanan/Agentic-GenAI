from langchain.agents import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
import requests, os, json
from bs4 import BeautifulSoup
from typing import Annotated, List, Dict, Any, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
import gradio as gr

load_dotenv()
Groq_api_key = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# @tool
# def internet_search(query: str):
#     """Searches the internet using DuckDuckGo."""
#     try:
#         with DDGS() as ddgs:
#             results = [r for r in ddgs.text(query, max_results=3)]
#             return results if results else "No results found."
#     except ValueError as e:
#         return f"Search failed: {str(e)}"
#     except Exception as e:
#         return f"An unexpected error occurred: {str(e)}"


@tool("internet_search", return_direct=False)
def internet_search(query: str) -> str:
    """Performs a Google search using Serper.dev and returns the top snippet."""
    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query
        }
        response = requests.post("https://google.serper.dev/search", json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract the top 3 snippets
        if "organic" in data and len(data["organic"]) > 0:
            results = []
            for result in data["organic"][:3]:  
                snippet = result.get("snippet", "No snippet found.")
                link = result.get("link", "")
                results.append(f"{snippet}\nLink: {link}")
    
            return "\n\n".join(results)
        else:
            return "No results found."

    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def process_content(url: str):
    """Processes content from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return f"Failed to retrieve content: {str(e)}"


tools = [internet_search, process_content]


# State
class AgentState(TypedDict):
    messages: List[Any]


# Initialize LLM
model = ChatGroq(groq_api_key=Groq_api_key, model_name="llama3-70b-8192")
model_with_tools = model.bind_tools(tools, tool_choice="any")


# -------------------------
# Agent Node: Call Model
# -------------------------
def call_model(state: AgentState) -> Dict:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


# -------------------------
# Response Node: Final Answer
# -------------------------
def respond(state: AgentState) -> Dict:
    last_message = state["messages"][-1]
    tool_call_response = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        tool_call_id = tool_call["id"]

        if tool_name == "internet_search":
            result = internet_search.invoke(args)
        elif tool_name == "process_content":
            result = process_content.invoke(args)
        else:
            result = f"Unknown tool: {tool_name}"

        tool_call_response.append(
            ToolMessage(tool_call_id=tool_call_id, content= result)
        )

    return {
        "final_response": result,
        "messages": [tool_call_response],
    }


# -------------------------
# Control Flow Logic
# -------------------------
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if (
        len(last_message.tool_calls) == 1
        and last_message.tool_calls[0]["name"]  in ["internet_search", "process_content"]
    ):
        return "respond"
    else:
        return "continue"


# Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "respond": "respond",
    },
)

workflow.add_edge("tools", "agent")
workflow.add_edge("respond", END)

graph = workflow.compile()


# Run Agent
# initial_input = {"messages": [HumanMessage(content="extract text from this url: https://medium.com/data-science/introducing-deep-learning-and-neural-networks-deep-learning-for-rookies-1-bd68f9cf5883")]}
# initial_input = {"messages": [HumanMessage(content="large language models")]}
# result = graph.invoke(initial_input)
# print(result['messages'][0][0].content)


# Gradio
def run_graph(input_message):
    response = graph.invoke({
        "messages": [HumanMessage(content=input_message)]
    })
    return json.dumps(response['messages'][0][0].content, indent=2)

inputs = gr.Textbox(lines=2, placeholder="Enter your query here...")
outputs = gr.Markdown(height=400, container=True)
title="LangGraph DuckDuckGo-Search"

demo = gr.Interface(fn=run_graph, inputs=inputs, outputs=outputs, title=title)
demo.launch(debug=True)