import autogen 
import os
import chromadb
from autogen import AssistantAgent,  UserProxyAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from IPython import get_ipython

os.environ["AUTOGEN_USE_DOCKER"] = "False"

config_list = [
    {
        "model": "llama3-70b-8192",
        "api_key": os.environ.get("GROQ_API_KEY"),
        "api_type": "groq",
    }
]


llm_config_proxy = {
    "temperature": 0.6,
    "config_list": config_list,
}


assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config_proxy,
    system_message="""You are a helpful assistant with retrieval power. Use the uploaded file content from llama-paper.pdf for generating accurate answers.Respond "Unsure about answer" if uncertain.""",
)

user = RetrieveUserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "code",
        "docs_path": ['llama-paper.pdf'],
        "chunk_token_size": 1000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path='/tmp/chromadb'),
        "collection_name": "pdfreader",
        "get_or_create": True,
    },
    code_execution_config={"work_dir": "coding"},
)

user_question = """
Compose a short blog post showcasing how Llama model is revolutionizing the future of Generative AI tarting with an engaging introduction that highlights its importance in generative AI. 
In the main body, cover its architecture, key features, and how it is transforming AI applications. Conclude with insights on its potential future impact and invite readers to share their thoughts. 
Keep the post under 500 words.
"""

user.initiate_chat(
    assistant,
    message=user_question,
)


