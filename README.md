# 🚀 Project 2: Agentic RAG with Learned Ranker (XGBoost / LightGBM + GPU Support)

## 📌 Overview

This project builds a **production-grade Retrieval-Augmented Generation (RAG) system** enhanced with a **learned ranker**.

Unlike vanilla RAG, this system:
- Retrieves candidates via vector similarity
- Re-ranks them using ML models
- Supports **classification and regression ranking**
- Includes **GPU acceleration (optional)**
- Is fully **testable, reproducible, and deployable**

---

## 🧠 Architecture

User Query → Embedder → Retriever → Feature Builder → Ranker → Answer → Critic

---

## ⚙️ Features

### 🔍 Retrieval
- FAISS-based similarity search
- SentenceTransformers embeddings

### 🧮 Learned Ranking
- Logistic / Linear
- XGBoost (GPU-ready)
- LightGBM (GPU-ready)

### ⚡ GPU Support
- Auto fallback to CPU if GPU unavailable

### 🧪 Testing
- Full pytest suite
- API + model + feature coverage

---

## 📁 Project Structure

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

---

## 🧾 Dataset Format

query,document,label,score

---

## 🛠️ Generate Dataset

python generate_training_pairs.py

---

## 🏋️ Train Ranker

python train_ranker.py

---

## ▶️ Run Pipeline

python main.py

---

## 🌐 Run API

uvicorn api:app --reload

Swagger: http://127.0.0.1:8000/docs

---

## 🧪 Run Tests

pytest -v

---

## 🧰 Install Dependencies

pip install fastapi uvicorn sentence-transformers faiss-cpu scikit-learn xgboost lightgbm pandas numpy joblib pytest httpx

---

## 🐳 Docker

docker build -t rag-project2 .
docker run -p 8000:8000 rag-project2

---

## ✅ Summary

- End-to-end RAG
- Learned ranking
- GPU support
- Full test coverage
