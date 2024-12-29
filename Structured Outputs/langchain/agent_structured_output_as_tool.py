### Bind structured output as a tool for LLM using pydantic
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os
import requests
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


load_dotenv()
weather_api_key = os.getenv('WEATHER_API_KEY')
if not weather_api_key:
    raise ValueError("WEATHER_API_KEY environment variable not set")

BASE_URL = "http://api.openweathermap.org/data/2.5/weather"


class WeatherResponse(BaseModel):
    """Structured response for weather information."""
    city: str = Field(description="The city name")
    temperature: float = Field(description="The temperature in Celsius")
    wind_directon: str = Field(description="The direction of the wind in abbreviated form")
    wind_speed: float = Field(description="The speed of the wind in m/s")


# Inherit 'messages' key from MessagesState, which is a list of chat messages
class AgentState(MessagesState):
    final_response: WeatherResponse  # Final structured response from the agent


def get_wind_direction(degrees):
    compass_points = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"
    ]
    index = round(degrees / 22.5) % 16
    return compass_points[index]


@tool
def get_weather(city: str): 
    """Fetch weather data for a given city."""
    params = {
        "q": city,
        "appid": weather_api_key,
        "units": "metric"  
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if response.status_code == 200:
        temperature = data['main']['temp']
        wind_speed = data['wind']['speed']
        city_name = data['name']
        wind_degrees = data['wind']['deg']
        wind_direction = get_wind_direction(wind_degrees)
        return f"The weather in {city_name} is {temperature}°C with wind speed {wind_speed} m/s coming from {wind_direction} direction."
    else:
        return f"Could not get the weather for {city}. Please try again."


Groq_api_key = os.getenv("GROQ_API_KEY")
if not Groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

model = ChatGroq(groq_api_key=Groq_api_key, model_name="llama3-8b-8192")


tools = [get_weather, WeatherResponse] 

# model determine the tools by passing tool_choice="any"
model_with_response_tool = model.bind_tools(tools, tool_choice="any")


# Define the function that calls the model
def call_model(state: AgentState):
    response = model_with_response_tool.invoke(state["messages"])
    return {"messages": [response]}


# Define the function that responds to the user with structured output
def respond(state: AgentState):
    weather_tool_call = state["messages"][-1].tool_calls[0]
    response = WeatherResponse(**weather_tool_call["args"])
    
    tool_message = {
        "type": "tool",
        "content": "Here is your structured response",
        "tool_call_id": weather_tool_call["id"],
    }
    
    return {"final_response": response, "messages": [tool_message]}  # final answer


# The function determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if (
        len(last_message.tool_calls) == 1
        and last_message.tool_calls[0]["name"] == "WeatherResponse"
    ):
        return "respond"
    else:
        return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", ToolNode(tools))

# Set the entrypoint as `agent`, this means that this node is the first one called
workflow.set_entry_point("agent")

# conditional edge
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


answer = graph.invoke(input={"messages": [("human", "what's the weather in Ratnapura?")]})[
    "final_response"
]

print(answer)
