import joblib

ml_models = {}

def load_ml_model(model_path="models/model.joblib"):
    return joblib.load(model_path)