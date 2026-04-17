import pandas as pd
from src.ranker_features import build_query_doc_features, make_feature_frame


def test_build_query_doc_features():
    query = "What is machine learning?"
    document = "Machine learning is a field of artificial intelligence."

    query_emb = [1.0, 0.0, 0.0]
    doc_emb = [1.0, 0.0, 0.0]

    feats = build_query_doc_features(query, document, query_emb, doc_emb)

    expected_keys = {
        "query_len",
        "doc_len",
        "len_diff",
        "cosine_sim",
        "shared_token_count",
    }

    assert expected_keys == set(feats.keys())
    assert feats["query_len"] == len(query)
    assert feats["doc_len"] == len(document)
    assert feats["cosine_sim"] == 1.0
    assert feats["shared_token_count"] >= 1


def test_make_feature_frame():
    rows = [
        {
            "query_len": 10,
            "doc_len": 100,
            "len_diff": 90,
            "cosine_sim": 0.9,
            "shared_token_count": 3,
        },
        {
            "query_len": 20,
            "doc_len": 80,
            "len_diff": 60,
            "cosine_sim": 0.5,
            "shared_token_count": 1,
        },
    ]

    df = make_feature_frame(rows)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == [
        "query_len",
        "doc_len",
        "len_diff",
        "cosine_sim",
        "shared_token_count",
    ]