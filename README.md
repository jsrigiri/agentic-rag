# 🚀 Agentic RAG System (End-to-End MLE + GenAI Project)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![RAG](https://img.shields.io/badge/Architecture-Agentic%20RAG-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

This project builds a **production-style Agentic Retrieval-Augmented Generation (RAG) system** that:

- Retrieves relevant documents using embeddings
- Generates answers using retrieved context
- Evaluates answers using a critic module
- Serves results via a low-latency API

---

## 🧠 Problem Statement

Large Language Models (LLMs) often **hallucinate** when they lack context.

This system addresses that by:

- Injecting **relevant knowledge** via retrieval
- Using **semantic search (FAISS)**
- Adding a **critic module for validation**
- Designing a **modular agent-based pipeline**

---

## 🏗 System Architecture

```text
User Query
   ↓
Embedding Model
   ↓
Vector Search (FAISS)
   ↓
Top-K Context Retrieval
   ↓
Agent (Answer Generation)
   ↓
Critic (Answer Validation)
   ↓
Final Response
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Embeddings | sentence-transformers |
| Vector Search | FAISS |
| API | FastAPI |
| Server | Uvicorn |
| Data Processing | NumPy |
| Validation | Pydantic |
| Testing | Pytest |

---

## 📂 Project Structure

```text
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
```

---

## 📊 Data Pipeline

- Documents stored in `data/documents.txt`
- Embedded using `sentence-transformers`
- Stored in FAISS index for similarity search

---

## 🔎 Retrieval System

- Converts query → embedding
- Performs vector similarity search
- Returns top-K relevant documents

---

## 🤖 Agent (Answer Generator)

- Combines retrieved context
- Generates answer using query + context

---

## 🧪 Critic (Answer Validator)

- Evaluates output quality
- Flags weak or short responses
- Enables future self-correction loops

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build vector store

```bash
python ingest.py
```

### 3. Run pipeline

```bash
python main.py
```

### 4. Start API

```bash
python -m uvicorn api:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 🔌 API Usage

### POST `/ask`

#### Input

```json
{
  "query": "What is machine learning?"
}
```

#### Output

```json
{
  "query": "What is machine learning?",
  "context": ["...", "..."],
  "answer": "Answer based on context...",
  "review": "Answer OK"
}
```

---

## 🧪 Testing

```bash
pytest
```

---

## 🔥 Key Highlights

- End-to-end RAG pipeline  
- FAISS-based vector search  
- Embedding-based semantic retrieval  
- Modular agent architecture  
- Critic evaluation layer  
- FastAPI deployment  

---

## 🚀 Future Improvements

- Integrate LLMs (OpenAI / Ollama / Claude)
- Add LangGraph orchestration
- Implement self-correction loop
- Add evaluation metrics
- Enable streaming responses
- Add memory / conversation context

---

## 📌 Author

Machine Learning Engineering Portfolio Project  
(GenAI + Systems + Production Focus)
