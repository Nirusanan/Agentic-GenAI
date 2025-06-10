from enum import Enum
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
import json
from typing import Optional

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

model = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

class Polarity(str, Enum):
    """
    An enumeration representing different levels of polarity.

    Attributes:
        positive (str): Represents the positive polarity.
        negative (str): Represents the negative polarity.
        neutral (str): Represents the neutral polarity.
    """
    positive = "Positive"
    negative = "Negative"
    neutral = "Neutral"


class CorePolarityExplanation(BaseModel):
    """
    core_polarity_explanation:
    - polarity: Sentiment polarity of the given sentence.
    - evidence: key phrases of the given sentence for supporting the polarity decision.
    """
    polarity: Polarity
    evidence: Optional[str] = None

sentences = [
    "I absolutely loved the movie; it was a masterpiece from start to finish!",

    "The customer service was excellent, and I felt truly valued.",

    "What a beautiful day—everything feels just right!",

    "I'm very disappointed with the product; it stopped working after one day.",

    "The food was cold, tasteless, and not worth the price at all.",

    "She always ignores me and makes me feel invisible.",

]



structured_llm = model.with_structured_output(CorePolarityExplanation)

for i in sentences:
    print(i)
    message = structured_llm.invoke(f"Find the polarity and sentiment key phrase for the given text: {i}")

    print("Polarity:", message.polarity.value)
    print("Key-Phrase Evidence:", message.evidence)  
    print()
