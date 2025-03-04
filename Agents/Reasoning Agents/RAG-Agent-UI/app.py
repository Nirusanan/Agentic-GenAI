import os
import streamlit as st
import autogen
import chromadb
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

api_key = os.getenv('GROQ_API_KEY')
model = "deepseek-r1-distill-qwen-32b" #deepseek-r1-distill-llama-70b

os.environ["AUTOGEN_USE_DOCKER"] = "False"

config_list = [
    {
    "model": model,
    "api_key": api_key,
    "api_type": "groq",
    }
]

llm_config = {
    "timeout": 60,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

URL = "research-paper/parametric-RAG.pdf"


def init_agent():

    boss_aid = RetrieveUserProxyAgent(
        name="Boss_Assistant",
        is_termination_msg=termination_msg,
        system_message="Assistant who has extra content retrieval power for solving difficult problems.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        retrieve_config={
            "task": "code",
            "docs_path": URL,
            "chunk_token_size": 1000,
            "model": config_list[0]["model"],
            "client": chromadb.PersistentClient(path="/tmp/chromadb"),
            "collection_name": "groupchat",
            "get_or_create": True,
        },
        code_execution_config=False,
    )

    coder = AssistantAgent(
        name="Senior_Python_Engineer",
        is_termination_msg=termination_msg,
        system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
        llm_config=llm_config,
    )

    groupchat = autogen.GroupChat(
        agents=[boss_aid, coder],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    return boss_aid, manager



st.title("Knowledge Agent Chat")

if "agent" not in st.session_state:
    st.session_state.agent, st.session_state.manager = init_agent()
    st.session_state.messages = []

if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_chat = st.session_state.agent.initiate_chat(recipient = st.session_state.manager ,message = prompt)
        response = response_chat.chat_history[1]["content"]
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})