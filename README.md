# 🚀 Agentic RAG System (ML Ranking + XGBoost + LightGBM + GPU)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/Models-XGBoost%20%7C%20LightGBM-orange)
![API](https://img.shields.io/badge/API-FastAPI-green)
![Tests](https://img.shields.io/badge/Tests-Pytest-blue)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

This project builds an **Agentic Retrieval-Augmented Generation (RAG) system** with a **learned ranking layer**.

It supports:

- Retrieval → FAISS + embeddings  
- Learned ranking → ML models  
- Classification → relevance (0/1)  
- Regression → ranking score  
- Multiple models:
  - Logistic / Linear (baseline)
  - XGBoost
  - LightGBM  
- Optional GPU acceleration  
- FastAPI inference API  
- Comprehensive pytest coverage  

---

## 🧠 Problem Statement

Improve retrieval quality in RAG systems:

- Retrieve candidate documents using embeddings  
- Learn to rank them using ML  
- Select best context for answer generation  
- Validate outputs using a critic  

---

## 🏗 Architecture

```text
User Query
   ↓
Embedding Model
   ↓
FAISS Retrieval
   ↓
Feature Engineering
   ↓
ML Ranker
   ↓
Top-K Context
   ↓
Answer Generation
   ↓
Critic Validation
```

---

## ⚙️ Tech Stack

| Layer              | Tools |
|-------------------|------|
| Retrieval          | FAISS, SentenceTransformers |
| Modeling           | Scikit-learn, XGBoost, LightGBM |
| API                | FastAPI |
| Testing            | Pytest |
| Data Processing    | Pandas, NumPy |

---

## 📂 Project Structure

```text
agentic-rag/
├── data/
│   └── training_pairs.csv
├── src/
│   ├── embedder.py
│   ├── retriever.py
│   ├── ranker_features.py
│   ├── ranker_model.py
│   ├── critic.py
├── artifacts/
│   ├── model.joblib
│   ├── features.joblib
├── vector_store/
│   └── store.pkl
├── tests/
│   ├── test_embedder.py
│   ├── test_retriever.py
│   ├── test_ranker_features.py
│   ├── test_ranker_model.py
│   ├── test_critic.py
│   ├── test_api.py
├── api.py
├── train_ranker.py
├── main.py
├── config.py
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🧠 Models Supported

### Regression
- Linear Regression  
- XGBoost Regressor  
- LightGBM Regressor  

### Classification
- Logistic Regression  
- XGBoost Classifier  
- LightGBM Classifier  

---

## ⚡ GPU Support

Optional GPU acceleration:

```python
USE_GPU = True
```

### Behavior
- Uses GPU if available  
- Falls back to CPU automatically  
- Works on all environments  

---

## 🧪 Testing (Pytest)

Run:

```bash
pytest -v
```

### Coverage

- Embedding generation  
- Retrieval logic  
- Feature engineering  
- Model training (CPU + GPU fallback)  
- API endpoints  

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Train ranker (required)

```bash
python train_ranker.py
```

---

### 3. Run pipeline

```bash
python main.py
```

---

### 4. Start API

```bash
python -m uvicorn api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🔌 API Usage (Swagger Examples)

Go to:

```
http://127.0.0.1:8000/docs
```

### Request

```json
{
  "query": "What is machine learning?"
}
```

---

### Expected Response

```json
{
  "query": "What is machine learning?",
  "task_type": "classification",
  "ranker_model_type": "xgboost_clf",
  "ranked_context": [
    {
      "document": "Machine learning is a field of AI.",
      "retrieval_distance": 0.21,
      "rank_score": 0.94
    }
  ],
  "answer": "Machine learning is a field of artificial intelligence...",
  "review": "Answer OK"
}
```

---

## 🔥 Key Highlights

- RAG with learned ranking  
- Classification + regression modeling  
- XGBoost & LightGBM integration  
- GPU-aware training with fallback  
- API deployment  
- Strong test coverage  

---

## 🧠 Talking Points

- Built agentic RAG system with ML-based ranking  
- Improved retrieval quality using learned signals  
- Compared linear vs boosting models  
- Implemented GPU-aware training  
- Designed modular pipeline for production  
- Added full test coverage for reliability  

---

## 📌 Author

Machine Learning + Quant + GenAI Portfolio Project
