# 🚀 Project 2: Agentic RAG with Learned Ranker

## 📌 Overview
A **production-grade Retrieval-Augmented Generation (RAG)** system with a **learned ranking layer**.

### 🔑 Key Capabilities
- 🔍 FAISS-based retrieval
- 🧠 ML-based re-ranking (XGBoost, LightGBM, Linear, Logistic)
- ⚡ GPU acceleration (optional with fallback)
- 🧪 Full pytest coverage
- 🌐 FastAPI deployment with Swagger

---

## 🧠 Architecture

```
User Query
   ↓
Embedder (SentenceTransformers)
   ↓
Retriever (FAISS)
   ↓
Feature Builder
   ↓
Ranker Model (ML)
   ↓
Top-K Context
   ↓
Answer Generator
   ↓
Critic (Quality Check)
```

---

## ⚙️ Features

### 🔍 Retrieval
- Dense embeddings using `sentence-transformers`
- Fast similarity search using FAISS

### 🧮 Learned Ranking
Supports both:

| Task | Models |
|------|--------|
| Classification | Logistic, XGBoost, LightGBM |
| Regression | Linear, XGBoost, LightGBM |

### ⚡ GPU Support
- XGBoost → `gpu_hist`
- LightGBM → `gpu` / `cuda`
- Automatic CPU fallback

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
├── data/
├── artifacts/
├── vector_store/
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
"What is ML?","Docker is container",0,0.10
```

- `label` → classification target  
- `score` → regression target  

---

## 🛠️ Generate Dataset

```bash
python generate_training_pairs.py
```

---

## 🏋️ Train Ranker

```bash
python train_ranker.py
```

### ⚙️ Config (`config.py`)

```python
TASK_TYPE = "classification"  # or "regression"
RANKER_MODEL_TYPE = "xgboost_clf"
USE_GPU = True
```

---

## ▶️ Run Pipeline

```bash
python main.py
```

---

## 🌐 Run API

```bash
uvicorn api:app --reload
```

### 📘 Swagger UI
http://127.0.0.1:8000/docs

---

## 📥 Example Request

```json
{
  "query": "What is machine learning?"
}
```

---

## 📤 Example Response

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

## 🧪 Run Tests

```bash
pytest -v
```

---

## 📦 Install Dependencies

```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu scikit-learn xgboost lightgbm pandas numpy joblib pytest httpx
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
- Regression enables finer ranking vs binary classification
- GPU support enables scalable training

---

## 🚀 Future Improvements

- Cross-encoder re-ranking
- Hybrid search (BM25 + embeddings)
- Online learning loop
- Distributed FAISS

---

## ✅ Summary

✔ End-to-end RAG pipeline  
✔ ML-based ranking  
✔ GPU acceleration  
✔ Full test coverage  
✔ Production-ready API  
