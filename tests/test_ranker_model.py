import numpy as np
import pytest
from src.ranker_model import train_ranker_model


def assert_valid_device_string(used_device: str):
    assert isinstance(used_device, str)
    assert (
        used_device == "cpu"
        or "gpu" in used_device
        or "fallback_cpu" in used_device
    )


def test_train_logistic_cpu(ranker_classification_xy):
    X, y = ranker_classification_xy
    model, used_device = train_ranker_model(
        X, y,
        model_type="logistic",
        use_gpu=False
    )

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert used_device == "cpu"


def test_train_linear_reg_cpu(ranker_regression_xy):
    X, y = ranker_regression_xy
    model, used_device = train_ranker_model(
        X, y,
        model_type="linear_reg",
        use_gpu=False
    )

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert used_device == "cpu"


def test_train_xgboost_clf_cpu(ranker_classification_xy):
    pytest.importorskip("xgboost")
    X, y = ranker_classification_xy
    model, used_device = train_ranker_model(
        X, y,
        model_type="xgboost_clf",
        use_gpu=False
    )

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert used_device == "cpu"


def test_train_xgboost_reg_cpu(ranker_regression_xy):
    pytest.importorskip("xgboost")
    X, y = ranker_regression_xy
    model, used_device = train_ranker_model(
        X, y,
        model_type="xgboost_reg",
        use_gpu=False
    )

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert used_device == "cpu"


def test_train_lightgbm_clf_cpu(ranker_classification_xy):
    pytest.importorskip("lightgbm")
    X, y = ranker_classification_xy
    model, used_device = train_ranker_model(
        X, y,
        model_type="lightgbm_clf",
        use_gpu=False
    )

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert used_device == "cpu"


def test_train_lightgbm_reg_cpu(ranker_regression_xy):
    pytest.importorskip("lightgbm")
    X, y = ranker_regression_xy
    model, used_device = train_ranker_model(
        X, y,
        model_type="lightgbm_reg",
        use_gpu=False
    )

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert used_device == "cpu"


def test_train_xgboost_clf_gpu_requested(ranker_classification_xy):
    pytest.importorskip("xgboost")
    X, y = ranker_classification_xy
    model, used_device = train_ranker_model(
        X, y,
        model_type="xgboost_clf",
        use_gpu=True
    )

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert_valid_device_string(used_device)


def test_train_xgboost_reg_gpu_requested(ranker_regression_xy):
    pytest.importorskip("xgboost")
    X, y = ranker_regression_xy
    model, used_device = train_ranker_model(
        X, y,
        model_type="xgboost_reg",
        use_gpu=True
    )

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert_valid_device_string(used_device)


def test_train_lightgbm_clf_gpu_requested(ranker_classification_xy):
    pytest.importorskip("lightgbm")
    X, y = ranker_classification_xy
    model, used_device = train_ranker_model(
        X, y,
        model_type="lightgbm_clf",
        use_gpu=True,
        lightgbm_gpu_backend="gpu"
    )

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert_valid_device_string(used_device)


def test_train_lightgbm_reg_gpu_requested(ranker_regression_xy):
    pytest.importorskip("lightgbm")
    X, y = ranker_regression_xy
    model, used_device = train_ranker_model(
        X, y,
        model_type="lightgbm_reg",
        use_gpu=True,
        lightgbm_gpu_backend="gpu"
    )

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert_valid_device_string(used_device)


def test_invalid_model_type(ranker_regression_xy):
    X, y = ranker_regression_xy

    try:
        train_ranker_model(X, y, model_type="not_real", use_gpu=False)
        assert False, "Expected ValueError for invalid model_type"
    except ValueError as e:
        assert "Unsupported model_type" in str(e)