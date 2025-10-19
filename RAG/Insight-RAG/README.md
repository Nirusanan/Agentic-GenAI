# RAG-Powered QA Engine

This project is a **Retrieval-Augmented Generation (RAG)** application that analyzes **blog articles and research papers** to answer user questions intelligently.  
It uses **OpenAI Embeddings**, **FAISS Vector Store**, and **LangChain** for semantic search and context-aware question answering.  
The interface is built using **Streamlit** for an interactive user experience.

---

## 🚀 Features

- 🧩 Convert documents into vector embeddings using **OpenAI embeddings**
- 📚 Store and retrieve vectors efficiently using **FAISS**
- 💬 Ask questions about your documents — get **contextual, AI-generated answers**
- ⚡ Built on **LangChain** for modular RAG pipeline
- 🌐 Simple web interface powered by **Streamlit**

---

## ⚙️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Nirusanan/Agentic-GenAI.git
   cd Agentic-GenAI/RAG/Insight-RAG
   ```
   
2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Your API Key**

   Create a .env file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   ```
   
---

## ▶️ Run the Application
```bash
streamlit run main.py
```

---
 <img width="1905" height="707" alt="insight-RAG" src="https://github.com/user-attachments/assets/af26f203-4eec-455f-b4aa-745600dec537" />
