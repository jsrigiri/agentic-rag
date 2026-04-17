# 🚀 Project 2: Agentic RAG + Learned Ranker

## 📌 Overview
Production-grade RAG pipeline with **ML-based re-ranking**.

- FAISS retrieval
- Learned ranking (XGBoost / LightGBM / sklearn)
- Classification + Regression
- GPU support (auto fallback)
- FastAPI + Swagger
- Full pytest coverage

---

## 🧠 Pipeline
```
Query → Embed → Retrieve → Features → Rank → Answer → Critic
```

---

## 📁 Structure
```
agentic-rag/
├── src/
├── tests/
├── data/
├── artifacts/
├── vector_store/
├── api.py
├── train_ranker.py
├── main.py
├── config.py
├── pytest.ini
```

---

## ⚙️ Setup

### Install
```
pip install fastapi uvicorn sentence-transformers faiss-cpu scikit-learn xgboost lightgbm pandas numpy joblib pytest httpx
```

### Train (required)
```
python train_ranker.py
```

### Run
```
python main.py
uvicorn api:app --reload
```

Swagger:
```
http://127.0.0.1:8000/docs
```

---

## 🧪 Tests
```
pytest -v
```

---

## ⚙️ Config
```python
TASK_TYPE = "classification"
RANKER_MODEL_TYPE = "xgboost_clf"
USE_GPU = True
```

---

## ❗ Notes
- Requires:
  - `artifacts/model.joblib`
  - `vector_store/store.pkl`
- GPU falls back to CPU automatically

---

## ✅ Summary
- RAG + learned ranking
- GPU-ready
- API + tests
- Production structured
