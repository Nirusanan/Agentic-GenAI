# ReAct agent
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
import requests
from langchain_core.tools import tool

load_dotenv()
Groq_api_key = os.getenv("GROQ_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

checkpointer = InMemorySaver()

@tool
def get_weather(city: str)-> str:
        """Get weather for a given city."""
        url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric'
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            city_name = city
            temperature = data['main']['temp']
            wind_speed = data['wind']['speed']
            humidity = data['main']['humidity']
          
            return f"The weather in {city_name} is {temperature}°C and {humidity} with wind speed {wind_speed} m/s."
        else:
            return f"Could not get the weather for {city_name}. Please try again."
    

agent = create_react_agent(
    model="groq:llama-3.1-8b-instant", 
    tools=[get_weather],
    checkpointer=checkpointer,
    prompt="You are a helpful assistant."  
)

# Run the agent
config = {"configurable": {"thread_id": "2"}}

sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Jaffna"}]},
    config  
)

tool_msgs = [m for m in sf_response["messages"] if m.type == "tool"]
if tool_msgs:
    print(tool_msgs[-1].content)
else:
    print(sf_response["messages"][-1].content)