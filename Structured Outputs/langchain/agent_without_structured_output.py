### Langchain Agent using Tool Calls without Structured Output
from dotenv import load_dotenv
import os
import requests
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from json import loads

# Load environment variables
load_dotenv()
weather_api_key = os.getenv("WEATHER_API_KEY")
if not weather_api_key:
    raise ValueError("WEATHER_API_KEY environment variable not set")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

BASE_URL = "http://api.openweathermap.org/data/2.5/weather"


def get_wind_direction(degrees):
    compass_points = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"
    ]
    index = round(degrees / 22.5) % 16
    return compass_points[index]


@tool
def get_weather(city: str) -> str:
    """
    Fetches the current weather for the specified city using the OpenWeather API.
    Args:
        city (str): The name of the city to fetch weather data for.
    Returns:
        str: A string describing the temperature, wind speed, and wind direction.
    """
    
    params = {
        "q": city,
        "appid": weather_api_key,
        "units": "metric"  
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if response.status_code == 200:
        temperature = data["main"]["temp"]
        wind_speed = data["wind"]["speed"]
        wind_degrees = data["wind"]["deg"]
        wind_direction = get_wind_direction(wind_degrees)

        return (
            f"The current weather in {city} is as follows:\n"
            f"Temperature: {temperature}°C\n"
            f"Wind Speed: {wind_speed} m/s\n"
            f"Wind Direction: {wind_direction}"
        )
    else:
        return f"Could not retrieve the weather for {city}. Error: {data.get('message', 'Unknown error')}"


model = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
tools = [get_weather]
model_with_tools = model.bind_tools(tools, tool_choice="any")

# State Management
class AgentState(MessagesState):
    pass


# define the call_model node
def call_model(state: AgentState):
    # Invoke the model with the current messages
    response = model_with_tools.invoke(state["messages"])

    if response.additional_kwargs.get("tool_calls"):
        tool_call = response.additional_kwargs["tool_calls"][0]
        tool_name = tool_call["function"]["name"]
        tool_args = tool_call["function"]["arguments"]

        # Execute the tool
        if tool_name == "get_weather":
            tool_arguments = loads(tool_args) 
            print("Tool Arguments:", tool_arguments)

            city = tool_arguments.get("city")  
            if city:  
                tool_result = get_weather.invoke({"city": city})

                if tool_result: 
                    # Add tool result as a valid assistant message
                    state["messages"].append({
                        "type": "assistant",
                        "content": tool_result
                    })
    return {"messages": state["messages"]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)  
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)  
graph = workflow.compile()

# Running the Graph
user_message = "What's the weather in Colombo?"
result = graph.invoke(input={"messages": [{"type": "human", "content": user_message}]})

print("Final Response:")
print(result["messages"][-1].content)
