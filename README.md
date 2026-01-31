# 🧠 Multi-Agent Research Assistant

**multi-agent GenAI research assistant** built using **FastAPI, LangGraph, ChromaDB, and Streamlit**.  
The system performs intelligent research using **web search, vector memory (RAG), and hybrid retrieval**, while streaming agent reasoning and responses in real time.

---

## 🚀 Key Features

- 🤖 **Multi-Agent Architecture**
  - Manager Agent (strategy & routing)
  - Research Agent (web + memory retrieval)
  - Validation Agent (fact extraction & filtering)
  - Summary Agent (final response generation)

- 🧠 **Intelligent Manager Agent**
  - Dynamically decides between:
    - Direct answer
    - Web research
    - Memory retrieval (ChromaDB)
    - Hybrid (Web + Memory)

- 📚 **RAG with ChromaDB**
  - Stores research results as embeddings
  - Reuses knowledge across queries
  - Reduces redundant web searches

- 🔀 **Hybrid Retrieval Strategy**
  - Combines historical knowledge with fresh web data

- 🔍 **Web Search Integration**
  - Uses DuckDuckGo (`ddgs`) for external research

- 📡 **Real-Time Streaming**
  - Streams agent steps and LLM tokens
  - Server-Sent Events (SSE) via FastAPI
  - Live UI updates in Streamlit

- 🪄 **Explainable AI**
  - Displays agent progress and reasoning steps
  - Transparent research workflow

---
