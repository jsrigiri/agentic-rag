from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


def _build_model(model_type: str, use_gpu: bool = False, lightgbm_gpu_backend: str = "gpu"):
    used_device = "cpu"

    if model_type == "logistic":
        return LogisticRegression(max_iter=1000), used_device

    if model_type == "linear_reg":
        return LinearRegression(), used_device

    if model_type == "xgboost_clf":
        params = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        if use_gpu:
            params["device"] = "cuda"
            used_device = "gpu"
        return XGBClassifier(**params), used_device

    if model_type == "xgboost_reg":
        params = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }
        if use_gpu:
            params["device"] = "cuda"
            used_device = "gpu"
        return XGBRegressor(**params), used_device

    if model_type == "lightgbm_clf":
        params = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }
        if use_gpu:
            params["device_type"] = lightgbm_gpu_backend
            used_device = f"gpu:{lightgbm_gpu_backend}"
        return LGBMClassifier(**params), used_device

    if model_type == "lightgbm_reg":
        params = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }
        if use_gpu:
            params["device_type"] = lightgbm_gpu_backend
            used_device = f"gpu:{lightgbm_gpu_backend}"
        return LGBMRegressor(**params), used_device

    raise ValueError(f"Unsupported model_type: {model_type}")


def train_ranker_model(X, y, model_type="logistic", use_gpu=False, lightgbm_gpu_backend="gpu"):
    model, intended_device = _build_model(
        model_type=model_type,
        use_gpu=use_gpu,
        lightgbm_gpu_backend=lightgbm_gpu_backend
    )

    try:
        model.fit(X, y)
        return model, intended_device
    except Exception as e:
        if use_gpu:
            cpu_model, cpu_device = _build_model(
                model_type=model_type,
                use_gpu=False,
                lightgbm_gpu_backend=lightgbm_gpu_backend
            )
            cpu_model.fit(X, y)
            return cpu_model, f"{intended_device} -> fallback_cpu ({type(e).__name__})"
        raise