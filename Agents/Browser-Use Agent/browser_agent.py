from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

async def main():
    agent = Agent(
        task = "Compare the price of damro refrigerator and abans refrigerator",
        llm = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.0-flash-exp")
    )
    await agent.run()

asyncio.run(main())