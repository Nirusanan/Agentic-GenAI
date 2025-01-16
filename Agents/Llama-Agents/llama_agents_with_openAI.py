from llama_agents import (
    AgentService,
    AgentOrchestrator,
    ControlPlaneServer,
    LocalLauncher,
    SimpleMessageQueue,
)

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.agent import FunctionCallingAgentWorker


# Calculator tool 
def calculator(operation: str, a: float, b: float) -> str:
    """
    Perform basic arithmetic operations.
    
    Args:
    operation (str): One of 'add', 'subtract', 'multiply', or 'divide'.
    a (float): First number.
    b (float): Second number.
    
    Returns:
    str: Result of the operation as a string.
    """
    try:
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return "Error: Cannot divide by zero."
            result = a / b
        else:
            return f"Error: Invalid operation '{operation}'. Choose 'add', 'subtract', 'multiply', or 'divide'."
        
        return f"The result of {a} {operation} {b} is {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

# Text analysis tool
def text_analyzer(text: str) -> str:
    """
    Perform basic text analysis.
    
    Args:
    text (str): The text to analyze.
    
    Returns:
    str: Analysis results as a string.
    """
    try:
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        analysis = f"Text Analysis Results:\n"
        analysis += f"- Word count: {word_count}\n"
        analysis += f"- Character count: {char_count}\n"
        analysis += f"- Approximate sentence count: {sentence_count}\n"
        
        return analysis
    except Exception as e:
        return f"Error in text analysis: {str(e)}"

calculator_tool = FunctionTool.from_defaults(fn=calculator)
text_tool = FunctionTool.from_defaults(fn=text_analyzer)



agent1 = ReActAgent.from_tools(tools=[calculator_tool], llm=OpenAI(), verbose=True)
agent2 = ReActAgent.from_tools(tools=[text_tool], llm=OpenAI(), verbose=True)


# worker1 = FunctionCallingAgentWorker.from_tools([calculator_tool], llm=OpenAI())
# agent1 = worker1.as_agent()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# create our multi-agent framework components
message_queue = SimpleMessageQueue()
control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=AgentOrchestrator(llm=OpenAI()),

)
agent_server_1 = AgentService(
    agent=agent1,
    message_queue=message_queue,
    description="Useful for performing basic arithmetic operations like calculations.",
    service_name="calculator_agent",
)
agent_server_2 = AgentService(
    agent=agent2,
    message_queue=message_queue,
    description="Useful for performing NLP, Text Analysis and Text Processing.",
    service_name="nlp_agent",
)

# launch
launcher = LocalLauncher([agent_server_1, agent_server_2], control_plane, message_queue)

try:
    result = launcher.launch_single("can you divide 100 by 20?")


    print(f"Result: {result}")
except Exception as e:
    print(f"An error occurred: {str(e)}")