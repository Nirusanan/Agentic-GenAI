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
Bayes Theorem is defined as $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$, representing the probability of event $A$ occurring given that $B$ is true. 
This formula relies on prior knowledge $P(A)$, the likelihood $P(B|A)$, and the evidence $P(B)$. 
In machine learning, it aids in classification tasks, such as spam detection. If $P(A)$ is high and $P(B|A)$ is significant, then $P(A|B)$ becomes meaningful. 
For example, predicting disease from symptoms uses $P(Disease|Symptoms)$. Bayes approach contrasts with frequentist methods by incorporating belief updates. 
As data grows, $P(A|B)$ converges to a more accurate value through repeated adjustments.
"""


class Latex(BaseModel):
    """
    Find all LaTeX-formatted words or expressions in the given text.
    Examples for LaTeX-formatted text: [$A$, $Large Language  Models$, $(a+b)**2 = a**2 + 2ab + b**2$]
    """
    latext_words: list[str]


structured_llm = model.with_structured_output(Latex)

response  = structured_llm.invoke(f"find all LaTeX words, sentences and equations from give text: {text}")
print(response)