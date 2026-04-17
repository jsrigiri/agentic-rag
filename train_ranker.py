import os
import joblib
import pickle
import pandas as pd

from config import (
    EMBEDDING_MODEL,
    TASK_TYPE,
    RANKER_MODEL_TYPE,
    USE_GPU,
    LIGHTGBM_GPU_BACKEND,
    ARTIFACTS_DIR,
    RANKER_MODEL_PATH,
    RANKER_FEATURES_PATH,
)
from src.embedder import Embedder
from src.ranker_features import build_query_doc_features, make_feature_frame
from src.ranker_model import train_ranker_model

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

df = pd.read_csv("data/training_pairs.csv")

embedder = Embedder(EMBEDDING_MODEL)

feature_rows = []
for _, row in df.iterrows():
    q_emb = embedder.encode([row["query"]])[0]
    d_emb = embedder.encode([row["document"]])[0]
    feats = build_query_doc_features(row["query"], row["document"], q_emb, d_emb)
    feature_rows.append(feats)

X = make_feature_frame(feature_rows)

if TASK_TYPE == "classification":
    y = df["label"]
elif TASK_TYPE == "regression":
    y = df["score"]
else:
    raise ValueError(f"Unsupported TASK_TYPE: {TASK_TYPE}")

model, used_device = train_ranker_model(
    X,
    y,
    model_type=RANKER_MODEL_TYPE,
    use_gpu=USE_GPU,
    lightgbm_gpu_backend=LIGHTGBM_GPU_BACKEND,
)

joblib.dump(model, RANKER_MODEL_PATH)
joblib.dump(list(X.columns), RANKER_FEATURES_PATH)

print("Saved ranker model to", RANKER_MODEL_PATH)
print("Saved ranker feature columns to", RANKER_FEATURES_PATH)
print("Task:", TASK_TYPE)
print("Model:", RANKER_MODEL_TYPE)
print("Device:", used_device)