from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
Groq_api_key = os.getenv("GROQ_API_KEY")

def book_hotel(hotel_name: str):
    """Book a hotel. Args: hotel_name (str): Name of the hotel to book."""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

flight_assistant = create_react_agent(
    model="groq:llama-3.1-8b-instant", 
    tools=[book_flight],
    prompt="You are a flight booking assistant",
    name="flight_assistant"
)

hotel_assistant = create_react_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[book_hotel],
    prompt="You are a hotel booking assistant",
    name="hotel_assistant"
)

supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=ChatGroq(model="llama-3.1-8b-instant"),
    prompt=(
        "You manage a flight booking assistant and a hotel booking assistant."
        "The only available tools are:\n"
        "- book_flight(from_airport, to_airport)\n"
        "- book_hotel(hotel_name)\n"
        "Always call these tools using their exact names and arguments."
    )
).compile()

for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "book a flight from Chennai to Delhi and a stay at TAJ Hotel"
            }
        ]
    }
):
    print(chunk)
    print("\n")