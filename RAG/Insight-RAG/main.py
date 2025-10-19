import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_openai import OpenAIEmbeddings


from dotenv import load_dotenv
load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("📚 Research Tool")
st.sidebar.title("Research Papers & Articles")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm=ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    vectorindex_openai = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    vectorindex_openai.save_local("faiss_index")

file_path = "faiss_index"
query = main_placeholder.text_input("💬 Ask a question about the loaded articles: ")


if query:
    if os.path.exists(file_path):
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        result = chain.invoke({"question": query}, return_only_outputs=True)

        st.header("🧠 Answer")
        # st.write(result["answer"])
        st.write(result.get("answer", "No answers generated"))

        sources = result.get("sources", "")
        if sources:
            st.subheader("🔗 Sources")
            sources_list = sources.split("\n")  
            for source in sources_list:
                st.write(source)