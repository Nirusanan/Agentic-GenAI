### use LLM that has structured output using `with_structured_output` method
import os
import requests
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()
weather_api_key = os.getenv("WEATHER_API_KEY")
if not weather_api_key:
    raise ValueError("WEATHER_API_KEY environment variable not set")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

BASE_URL = "http://api.openweathermap.org/data/2.5/weather"


class WeatherResponse(BaseModel):
    """Structured response for weather information."""
    city: str = Field(description="The city name")
    temperature: float = Field(description="The temperature in Celsius")
    wind_directon: str = Field(description="The direction of the wind in abbreviated form")
    wind_speed: float = Field(description="The speed of the wind in m/s")


class AgentState(MessagesState):
    final_response: WeatherResponse  


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


model = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
tools = [get_weather]
model_with_tools = model.bind_tools(tools, tool_choice="any")
model_with_structured_output = model.with_structured_output(WeatherResponse)


# Define the function that calls the model
def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Define the function that responds to the user
def respond(state: AgentState):
    # We call the model with structured output in order to return the same format to the user every time
    # state['messages'][-2] is the last ToolMessage in the convo, which we convert to a HumanMessage for the model to use
    response = model_with_structured_output.invoke(
        [HumanMessage(content=state["messages"][-2].content)]
    )
    
    return {"final_response": response}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    print(f"Debug: Last message tool_calls - {last_message.tool_calls}")

    if len(last_message.tool_calls)==1:
        return "respond"
    else:
        return "continue"
    # if (
    #     len(last_message.tool_calls) == 1
    #     and last_message.tool_calls[0]["name"] == "get_weather"
    # ):
    #     return "respond"
    # else:
    #     return "continue"



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

answer = graph.invoke(input={"messages": [("human", "what's the weather in Colombo?")]})[
    "final_response"
]

print(answer)