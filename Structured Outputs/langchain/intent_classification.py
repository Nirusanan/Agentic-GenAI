from enum import Enum
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
import json
from langchain_core.prompts import PromptTemplate

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

model = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

class Severity(str, Enum):
    """
    An enumeration representing different levels of severity.

    Attributes:
        low (str): Represents a low level of severity.
        medium (str): Represents a medium level of severity.
        high (str): Represents a high level of severity.
    """
    low = "Low"
    medium = "Medium"
    high = "High"


class CoreProblem(BaseModel):
    """
    core_problem (str): Represent the main core problems keywords of the given complaint
    """
    problem: str


class Result(BaseModel):
    severity: Severity
    core_problem: list[CoreProblem]


complaints = [
    "I've been experiencing frequent dropped calls in my area, making it hard to stay connected with important contacts. This issue has been ongoing for weeks without any resolution.",

    "The mobile data speed is significantly slower than what was promised in the contract. Streaming videos or even simple web browsing has become frustratingly sluggish.",

    "I'm disappointed with the customer service responsiveness. Every time I call for help, there is a long wait time and sometimes representatives are unable to resolve my issues satisfactorily.",

    "Despite being enrolled in an unlimited data plan, I keep getting charged for data overages. I've tried reaching out for clarification but haven't received a clear answer.",

    "The mobile app is often unresponsive and crashes frequently. It's difficult to manage my account or check my data usage with such an unreliable tool.",

    "Roaming charges are exorbitantly high and not communicated clearly beforehand. I was shocked by my bill after a recent international trip.",

    "My phone's signal frequently drops even when I'm in urban areas. This is causing significant inconvenience and making me reconsider my current provider.",

    "There was an unexpected charge on my bill that customer service insists is legitimate, but I have no record of incurring such costs. Getting it rectified has been a hassle.",

    "The promotional offers are misleading and often come with hidden terms that are not initially disclosed. It feels like I'm being deceived rather than rewarded.",

    "Network congestion frequently results in call failures during peak hours. I expect better performance given the competitive rates in the market."
]


prompt = PromptTemplate.from_template("Find the severity and core problem for the given text: {complaint}")
structured_llm = model.with_structured_output(Result)

for i in complaints:
    print(i)
    prompt_input = prompt.format(complaint=i)  #formated string
    result = structured_llm.invoke(prompt_input)
    # result = structured_llm.invoke(f"Find the severity and core problem for the given text: {i}")
    
    print("Severity:", result.severity.value)
    print("Core Problem:", result.core_problem[0].problem)  
    print()
