# 🚀 Agentic RAG System (End-to-End MLE + GenAI Project)

An end-to-end **Agentic Retrieval-Augmented Generation (RAG)** system that retrieves relevant documents, generates answers, and evaluates outputs using a critic module — all served via a production-ready API.

---

## 📌 Problem

Traditional LLMs hallucinate when they lack context.  
This project solves that by:

- Retrieving relevant documents from a knowledge base
- Generating answers using retrieved context
- Evaluating answers using a critic module
- Serving the system via an API

---

## 🏗 Architecture

User Query
   ↓
Embedding Model
   ↓
Vector Search (FAISS)
   ↓
Context Retrieval
   ↓
Agent (Answer Generation)
   ↓
Critic (Answer Validation)
   ↓
Final Response

---

## ⚙️ Tech Stack

- Embeddings: sentence-transformers  
- Vector Search: FAISS  
- API: FastAPI  
- Server: Uvicorn  
- Data Processing: NumPy  
- Validation: Pydantic  
- Testing: Pytest  

---

## 📂 Project Structure

agentic-rag/
├── data/
│   └── documents.txt
├── src/
│   ├── embedder.py
│   ├── retriever.py
│   ├── agent.py
│   ├── critic.py
│   └── utils.py
├── vector_store/
│   └── store.pkl
├── ingest.py
├── main.py
├── api.py
├── config.py
├── requirements.txt
├── tests/
└── README.md

---

## 📊 Data Pipeline

- Documents stored in data/documents.txt
- Embedded using sentence-transformers
- Stored in FAISS index for similarity search

---

## 🔎 Retrieval System

- Converts query → embedding
- Finds top-K similar documents
- Returns relevant context

---

## 🤖 Agent (Answer Generator)

- Combines retrieved context
- Generates response based on query + context

---

## 🧪 Critic (Answer Validator)

- Evaluates response quality
- Flags short or weak answers
- Enables future self-correction loops

---

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt

---

### 2. Build vector store
python ingest.py

---

### 3. Run pipeline
python main.py

---

### 4. Run API
python -m uvicorn api:app --reload

Open:
http://127.0.0.1:8000/docs

---

## 🔌 API Usage

### POST /ask

Input:
{
  "query": "What is machine learning?"
}

Output:
{
  "query": "...",
  "context": ["...", "..."],
  "answer": "...",
  "review": "Answer OK"
}

---

## 🧪 Tests

pytest

---

## 📊 Example Output

- Retrieved documents relevant to query
- Generated answer using context
- Critic feedback on answer quality

---

## 🔥 Key Highlights (MLE + GenAI Focus)

- End-to-end RAG pipeline  
- Vector database (FAISS) integration  
- Embedding-based semantic search  
- Agent-style architecture  
- Critic module for evaluation  
- API deployment with FastAPI  

---

## 🚀 Future Improvements

- Integrate LLMs (OpenAI / Ollama / Claude)
- Add multi-agent orchestration (LangGraph)
- Implement self-correction loop
- Add evaluation metrics
- Stream responses
- Add memory and conversation history

---

## 🧠 Interview Talking Points

- Built an Agentic RAG system with retrieval + reasoning
- Implemented embedding-based semantic search
- Designed modular architecture (retriever + agent + critic)
- Deployed system as production-ready API
- Demonstrated GenAI + MLE integration

---

## 📌 Author

Machine Learning Engineering Portfolio Project
