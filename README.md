# 🚀 Project 2: Agentic RAG with Learned Ranker (Verified Setup)

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system with a **learned ranking layer** using ML models.

It includes:
- FAISS retrieval
- ML-based ranking (classification + regression)
- GPU support (with fallback)
- FastAPI serving
- Full pytest coverage

---

## 📁 Project Structure (Verified)

```
agentic-rag/
├── src/
│   ├── embedder.py            # SentenceTransformer embeddings
│   ├── retriever.py           # FAISS search
│   ├── ranker_features.py     # Feature engineering
│   ├── ranker_model.py        # ML models (XGB, LGBM, etc.)
│   ├── critic.py              # Output validation
│
├── tests/                     # Pytest suite
│
├── data/
│   └── training_pairs.csv     # Training dataset
│
├── artifacts/
│   ├── model.joblib          # Trained ranker
│   ├── features.joblib       # Feature list
│
├── vector_store/
│   └── store.pkl             # FAISS embeddings
│
├── api.py                    # FastAPI app
├── train_ranker.py           # Train ranker
├── main.py                   # Run pipeline
├── config.py                 # Configurations
├── pytest.ini
```

---

## ⚠️ Important: Required Files Before Running

You MUST have:

1. Vector store:
```
vector_store/store.pkl
```

2. Trained model:
```
artifacts/model.joblib
artifacts/features.joblib
```

---

## 🛠️ Step-by-Step: How to Run

### 1️⃣ Install dependencies

```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu scikit-learn xgboost lightgbm pandas numpy joblib pytest httpx
```

---

### 2️⃣ Generate dataset (if not present)

```bash
python generate_training_pairs.py
```

---

### 3️⃣ Train ranker (CRITICAL STEP)

```bash
python train_ranker.py
```

This creates:
```
artifacts/model.joblib
artifacts/features.joblib
```

---

### 4️⃣ Build vector store (if not already created)

Make sure your pipeline creates:

```
vector_store/store.pkl
```

If missing, your API will fail.

---

### 5️⃣ Run pipeline (sanity check)

```bash
python main.py
```

---

### 6️⃣ Run API

```bash
uvicorn api:app --reload
```

Open Swagger:
```
http://127.0.0.1:8000/docs
```

---

## 🧪 Run Tests

```bash
pytest -v
```

---

## ❗ Common Errors & Fixes

### ❌ No such file: artifacts/model.joblib
➡️ Run:
```bash
python train_ranker.py
```

---

### ❌ No module named 'src'
➡️ Ensure `pytest.ini` contains:
```
[pytest]
pythonpath = .
```

---

### ❌ Empty ranked_context
➡️ Check:
- vector_store exists
- embeddings dimension matches model

---

### ❌ GPU not working
➡️ Expected behavior:
- Falls back to CPU automatically

---

## ⚙️ Config Example

```python
TASK_TYPE = "classification"   # or regression
RANKER_MODEL_TYPE = "xgboost_clf"
USE_GPU = True
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
  "ranked_context": [...],
  "answer": "...",
  "review": "..."
}
```

---

## ✅ Summary

✔ Verified project structure  
✔ Clear run sequence  
✔ Handles common failures  
✔ Ready for production/API use  
✔ Fully testable  

---

## 🚀 Next Step

Move to **Project 3: Full MLOps (CI/CD + Docker + Monitoring)**
