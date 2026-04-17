EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

TASK_TYPE = "classification"  
# classification or regression

RANKER_MODEL_TYPE = "xgboost_clf"
# logistic, xgboost_clf, lightgbm_clf, linear_reg, xgboost_reg, lightgbm_reg

USE_GPU = True
LIGHTGBM_GPU_BACKEND = "gpu"

ARTIFACTS_DIR = "artifacts"
RANKER_MODEL_PATH = "artifacts/ranker_model.joblib"
RANKER_FEATURES_PATH = "artifacts/ranker_feature_columns.joblib"
STORE_PATH = "vector_store/store.pkl"