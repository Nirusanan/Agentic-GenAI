import os
import requests
import yfinance as yf
from swarm import Swarm, Agent
import gradio as gr


client = Swarm()

API_KEY = os.getenv('OPENWEATHER_API_KEY')
if not API_KEY:
    raise ValueError("OPENWEATHER_API_KEY environment variable not set")

BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Fetch weather data
def get_weather(location):
    print(f"Running weather function for {location}...")
    
    params = {
        "q": location,
        "appid": API_KEY,
        "units": "metric"  
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if response.status_code == 200:
        temperature = data['main']['temp']
        weather_description = data['weather'][0]['description']
        city_name = data['name']
        return f"The weather in {city_name} is {temperature}°C with {weather_description}."
    else:
        return f"Could not get the weather for {location}. Please try again."

# Fetch stock price using yfinance
def get_stock_price(ticker):
    print(f"Running stock price function for {ticker}...")
    stock = yf.Ticker(ticker)
    stock_info = stock.history(period="1d")
    if not stock_info.empty:
        latest_price = stock_info['Close'].iloc[-1]
        return f"The latest stock price for {ticker} is {latest_price}."
    else:
        return f"Could not retrieve stock price for {ticker}."

# Function to transfer from manager agent to weather agent
def transfer_to_weather_assistant():
    print("Transferring to Weather Assistant...")
    return weather_agent

# Function to transfer from manager agent to stock price agent
def transfer_to_stockprice_assistant():
    print("Transferring to Stock Price Assistant...")
    return stockprice_agent

# Agents
manager_agent = Agent(
    name="manager Assistant",
    instructions="You help users by directing them to the right assistant.",
    functions=[transfer_to_weather_assistant, transfer_to_stockprice_assistant],
)

weather_agent = Agent(
    name="Weather Assistant",
    instructions="You provide weather information for a given location using the provided tool",
    functions=[get_weather],
)

stockprice_agent = Agent(
    name="Stock Price Assistant",
    instructions="You provide the latest stock price for a given ticker symbol using the yfinance library.",
    functions=[get_stock_price],
)


#  Gradio interface
prompt = gr.Textbox(label="Enter prompt (e.g., 'What's the weather in New York?', 'Get me the stock price of AAPL'):")

def execute_agent(prompt):
    response = client.run(
        agent=manager_agent,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.messages[-1]["content"]

iface = gr.Interface(
    fn=execute_agent, 
    inputs=prompt, 
    outputs="text",
    title="Agents for the Weather and Stock price"
)
iface.launch()