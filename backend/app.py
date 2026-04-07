from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Import your modules (based on YOUR structure)
from model.train import train_model
from bias.metrics import calculate_bias
from bias.mitigation import mitigate_bias
from explain.explain import get_feature_importance

app = FastAPI(title="AI Bias Audit API")

# Enable CORS (frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store dataset globally (simple hackathon approach)
DATA = {}
MODEL = {}

# ✅ Root check
@app.get("/")
def home():
    return {"message": "AI Bias Audit Backend Running"}

# ✅ Upload dataset
@app.post("/upload")
def upload_dataset():
    # TEMP: load from local file (simplest for hackathon)
    df = pd.read_csv("../data/sample_dataset.csv")
    DATA["df"] = df
    return {"message": "Dataset loaded successfully", "columns": list(df.columns)}

# ✅ Train model
@app.post("/train")
def train():
    df = DATA.get("df")
    if df is None:
        return {"error": "No dataset loaded"}

    model, X_test, y_test = train_model(df)
    MODEL["model"] = model
    MODEL["X_test"] = X_test
    MODEL["y_test"] = y_test

    return {"message": "Model trained successfully"}

# ✅ Bias detection
@app.get("/bias")
def bias_check():
    model = MODEL.get("model")
    X_test = MODEL.get("X_test")
    y_test = MODEL.get("y_test")

    if model is None:
        return {"error": "Model not trained"}

    result = calculate_bias(model, X_test, y_test)
    return result

# ✅ Bias mitigation
@app.post("/mitigate")
def mitigate():
    df = DATA.get("df")
    if df is None:
        return {"error": "No dataset"}

    new_result = mitigate_bias(df)
    return new_result

# ✅ Explainability
@app.get("/explain")
def explain():
    model = MODEL.get("model")
    if model is None:
        return {"error": "Model not trained"}

    explanation = get_feature_importance(model)
    return explanation