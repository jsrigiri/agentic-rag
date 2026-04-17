# 🚀 Project 2: Agentic RAG with Learned Ranker

---

## 📌 Overview

This project builds a **production-grade Retrieval-Augmented Generation (RAG)** system enhanced with a **learned ranking layer**, following the same structured style as Project 1.

The system:
- Retrieves candidate documents using embeddings + FAISS
- Re-ranks them using ML models (classification or regression)
- Supports **XGBoost, LightGBM, and sklearn models**
- Includes **GPU acceleration with safe fallback**
- Exposes a **FastAPI service with Swagger UI**
- Is fully covered with **pytest-based tests**

---

## 🧠 End-to-End Pipeline

```
User Query
   ↓
Embedder (SentenceTransformer)
   ↓
Retriever (FAISS)
   ↓
Feature Engineering
   ↓
Ranker Model (ML)
   ↓
Top-K Context
   ↓
Answer Generator
   ↓
Critic (Validation)
```

---

## ⚙️ Core Features

### 🔍 Retrieval Layer
- SentenceTransformers embeddings
- FAISS similarity search
- Top-K document retrieval

### 🧮 Learned Ranking Layer
Supports both:

| Mode | Models |
|------|--------|
| Classification | Logistic, XGBoost, LightGBM |
| Regression | Linear, XGBoost, LightGBM |

### ⚡ GPU Support
- XGBoost → `gpu_hist`
- LightGBM → `gpu` / `cuda`
- Automatic fallback to CPU if GPU unavailable

### 🧪 Testing
- Full pytest suite
- Covers API, ranking, features, retrieval, critic

---

## 📁 Project Structure

```
agentic-rag/
├── src/
│   ├── embedder.py
│   ├── retriever.py
│   ├── ranker_features.py
│   ├── ranker_model.py
│   ├── critic.py
│
├── tests/
│   ├── test_embedder.py
│   ├── test_retriever.py
│   ├── test_ranker_features.py
│   ├── test_ranker_model.py
│   ├── test_critic.py
│   ├── test_api.py
│
├── data/
│   └── training_pairs.csv
│
├── artifacts/
│   ├── model.joblib
│   ├── features.joblib
│
├── vector_store/
│   └── store.pkl
│
├── api.py
├── train_ranker.py
├── main.py
├── config.py
├── pytest.ini
```

---

## 🧾 Dataset Format

```
query,document,label,score
"What is ML?","Machine learning is AI",1,0.95
```

- `label` → classification target  
- `score` → regression target  

---

## 🛠️ Setup & Run (Step-by-Step)

### 1️⃣ Install Dependencies

```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu scikit-learn xgboost lightgbm pandas numpy joblib pytest httpx
```

---

### 2️⃣ Generate Dataset (Optional)

```bash
python generate_training_pairs.py
```

---

### 3️⃣ Train Ranker (Required)

```bash
python train_ranker.py
```

Creates:

```
artifacts/model.joblib
artifacts/features.joblib
```

---

### 4️⃣ Ensure Vector Store Exists

```
vector_store/store.pkl
```

If missing, retrieval will fail.

---

### 5️⃣ Run Full Pipeline

```bash
python main.py
```

---

### 6️⃣ Run API

```bash
uvicorn api:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 📘 API Example

### Request

```json
{
  "query": "What is machine learning?"
}
```

### Response

```json
{
  "query": "...",
  "task_type": "...",
  "ranker_model_type": "...",
  "ranked_context": [...],
  "answer": "...",
  "review": "..."
}
```

---

## 🧪 Running Tests

```bash
pytest -v
```

Run specific:

```bash
pytest tests/test_api.py -v
```

---

## ❗ Common Issues & Fixes

### Missing model.joblib
```bash
python train_ranker.py
```

### No module named 'src'
Ensure `pytest.ini`:

```
[pytest]
pythonpath = .
```

### Empty ranked_context
- Check vector store exists
- Check embeddings dimension consistency

### GPU not used
- Falls back to CPU automatically (expected behavior)

---

## ⚙️ Configuration

```python
TASK_TYPE = "classification"   # or "regression"
RANKER_MODEL_TYPE = "xgboost_clf"
USE_GPU = True
```

---

## 🐳 Docker (Optional)

```bash
docker build -t rag-project2 .
docker run -p 8000:8000 rag-project2
```

---

## 🧠 Design Highlights

- Learned ranking improves relevance vs cosine similarity
- Regression enables fine-grained ranking scores
- GPU support enables scalability
- Modular design for production systems

---

## 🚀 Future Improvements

- Cross-encoder re-ranking (BERT)
- Hybrid search (BM25 + embeddings)
- Online feedback loop
- Distributed FAISS

---

## ✅ Summary

✔ End-to-end RAG system  
✔ Learned ranking layer  
✔ GPU-enabled training  
✔ Full pytest coverage  
✔ Production-ready API  

---

## ➡️ Next Step

👉 Project 3: Full MLOps (CI/CD + Docker + Monitoring)
