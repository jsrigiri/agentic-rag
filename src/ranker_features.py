import numpy as np
import pandas as pd


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_query_doc_features(query, document, query_emb, doc_emb):
    return {
        "query_len": len(query),
        "doc_len": len(document),
        "len_diff": abs(len(query) - len(document)),
        "cosine_sim": cosine_similarity(query_emb, doc_emb),
        "shared_token_count": len(set(query.lower().split()) & set(document.lower().split())),
    }


def make_feature_frame(feature_dicts):
    return pd.DataFrame(feature_dicts)