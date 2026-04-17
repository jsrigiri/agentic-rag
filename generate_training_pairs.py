import random
from pathlib import Path
import pandas as pd

random.seed(42)

queries_to_positive_docs = {
    "What is machine learning?": [
        "Machine learning is a field of artificial intelligence that focuses on building systems that learn from data.",
        "Supervised learning involves training a model using labeled data.",
        "Unsupervised learning deals with unlabeled data using clustering and dimensionality reduction.",
        "Reinforcement learning is a paradigm where an agent learns by interacting with an environment.",
        "Feature engineering is the process of transforming raw data into features that improve model performance.",
    ],
    "What is FastAPI?": [
        "FastAPI is a modern, high-performance Python web framework used to build APIs.",
        "Uvicorn is an ASGI server used to run FastAPI applications.",
        "Model deployment is the process of making machine learning models available for inference via APIs or services.",
    ],
    "What is ridge regression?": [
        "Ridge regression is a regularization technique used in linear models. It adds an L2 penalty term to reduce overfitting and handle multicollinearity.",
        "Overfitting occurs when a model learns noise instead of the underlying pattern, leading to poor generalization.",
        "Hyperparameter tuning is the process of optimizing model parameters to improve performance.",
    ],
    "What is RAG?": [
        "Retrieval-Augmented Generation combines retrieval of relevant documents with language model generation to produce accurate and context-aware answers.",
        "Vector databases store embeddings of documents and allow efficient similarity search.",
        "Embeddings are numerical representations of text that capture semantic meaning.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    ],
    "What is MLflow?": [
        "MLflow is used for experiment tracking, model logging, and lifecycle management in machine learning systems.",
        "MLOps is a set of practices that combines machine learning, DevOps, and data engineering to automate and manage the ML lifecycle.",
        "CI/CD pipelines automate testing, building, and deployment of applications to ensure reliability and scalability.",
    ],
    "What is FAISS?": [
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "Vector databases store embeddings of documents and allow efficient similarity search.",
        "Embeddings are numerical representations of text that capture semantic meaning.",
    ],
    "What is backtesting?": [
        "Backtesting is the process of evaluating a trading strategy using historical data.",
        "Transaction costs play a critical role in trading strategies because ignoring costs can overestimate profitability.",
        "Lead-lag relationships occur when one asset's price movement precedes another in financial markets.",
    ],
    "What is Docker?": [
        "Docker is a containerization platform that allows applications to run consistently across environments.",
        "CI/CD pipelines automate testing, building, and deployment of applications to ensure reliability and scalability.",
        "Model deployment is the process of making machine learning models available for inference via APIs or services.",
    ],
    "What is Kafka?": [
        "Kafka is a distributed streaming platform used for building real-time data pipelines.",
        "Streaming data systems process data in real time, enabling low-latency applications such as trading and monitoring systems.",
        "Latency is a critical factor in real-time systems and refers to the delay between input and output.",
    ],
    "What is time series data?": [
        "Time series data consists of observations indexed by time and is commonly used in finance, forecasting, and signal processing.",
        "Autoregressive models use past values of a variable to predict future values.",
        "Lead-lag relationships are common in financial markets.",
    ],
}

all_docs = sorted({doc for docs in queries_to_positive_docs.values() for doc in docs} | {
    "Large Language Models like GPT, Claude, and LLaMA can generate human-like text and are often used in RAG systems.",
    "Agentic AI systems involve planners, executors, and critics working together to solve complex tasks.",
    "A critic module in an AI system evaluates the quality of generated outputs.",
    "Scalability refers to a system's ability to handle increasing workloads efficiently.",
    "LightGBM and XGBoost are gradient boosting libraries widely used for tabular machine learning.",
    "Classification predicts discrete categories, while regression predicts continuous values.",
    "GPU acceleration can speed up training for boosting models when the environment supports it.",
    "Pytest is commonly used for Python unit and integration testing.",
})

rows = []
for query, positives in queries_to_positive_docs.items():
    for doc in positives:
        base_score = random.uniform(0.86, 0.99)
        rows.append({"query": query, "document": doc, "label": 1, "score": round(base_score, 2)})
        if "used to" in doc:
            aug = doc.replace("used to", "often used to")
        elif "is a" in doc:
            aug = doc.replace("is a", "is a widely used")
        else:
            aug = doc + " This concept is important in production ML systems."
        aug_score = max(0.8, min(0.99, base_score - random.uniform(0.01, 0.08)))
        rows.append({"query": query, "document": aug, "label": 1, "score": round(aug_score, 2)})

    negative_pool = [d for d in all_docs if d not in positives]
    sampled_negatives = random.sample(negative_pool, k=min(10, len(negative_pool)))
    for doc in sampled_negatives:
        score = random.uniform(0.02, 0.35)
        rows.append({"query": query, "document": doc, "label": 0, "score": round(score, 2)})

df = pd.DataFrame(rows).drop_duplicates().sample(frac=1, random_state=42).reset_index(drop=True)

out_path = Path("data/training_pairs.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
