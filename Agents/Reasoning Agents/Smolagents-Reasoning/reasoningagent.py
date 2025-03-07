import os
import litellm
from smolagents import LiteLLMModel

api_key = os.environ.get('GROQ_API_KEY')

model = LiteLLMModel(
    model_id="groq/deepseek-r1-distill-qwen-32b",
)


from smolagents import CodeAgent

agent = CodeAgent(
    tools=[],
    model=model,
)

agent.run("Could you give me the 18th number in the Fibonacci sequence?")
