### Find the category and confidence for each  category
from enum import Enum
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

model = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")


text = """
A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world. 
Konstantin Batygin did not set out to solve one of the solar system’s most puzzling mysteries when he went for a run up a hill in Nice, France. 
Dr. Batygin, a Caltech researcher, best known for his contributions to the search for the solar system’s missing “Planet Nine,” spotted a beer bottle. 
At a steep, 20 degree grade, he wondered why it wasn’t rolling down the hill. He realized there was a breeze at his back holding the bottle in place. 
Then he had a thought that would only pop into the mind of a theoretical astrophysicist: “Oh! This is how Europa formed.” 
Europa is one of Jupiter’s four large Galilean moons. And in a paper published Monday in the Astrophysical Journal, Dr. Batygin and a co-author, Alessandro Morbidelli, 
a planetary scientist at the Côte d’Azur Observatory in France, present a theory explaining how some moons form around gas giants like Jupiter and Saturn, 
suggesting that millimeter-sized grains of hail produced during the solar system’s formation became trapped around these massive worlds,
taking shape one at a time into the potentially habitable moons we know today."""

categories = ['space & cosmos', 'scientific discovery', 'microbiology', 'robots', 'archeology']

prompt = f"""
text: {text}

categories: {'| '.join(categories)}
"""

print(prompt)

class Category(BaseModel):
    """
    Given the text generate find the suitable category with confidence
    Include all categories with confidence 0 - 1
    The total confidence sum should be 1
    The confidence should be correct to the relevant text
    """
    category: str
    confidence: float

class Result(BaseModel):
    result: list[Category]


structured_llm = model.with_structured_output(Result)

response  = structured_llm.invoke(f"Find the category and confidence for the given prompt: {prompt}")
# print(response)

for item in response.result:
    print("Category:", item.category)
    print("Confidence:", item.confidence)
    print()
