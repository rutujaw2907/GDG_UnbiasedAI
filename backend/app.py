from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from model.train import train_model
from bias.metrics import calculate_bias
from bias.mitigation import mitigate_bias
from explain.explain import get_feature_importance

app = FastAPI(title="AI Bias Audit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA = {}
MODEL = {}

@app.get("/")
def home():
    return {"message": "AI Bias Audit Backend Running"}

@app.post("/upload")
def upload_dataset():
    df = pd.read_csv("../data/sample_dataset.csv")
    DATA["df"] = df
    return {"message": "Dataset loaded", "columns": list(df.columns)}

@app.post("/train")
def train():
    df = DATA.get("df")
    if df is None:
        return {"error": "No dataset loaded"}

    model, X_test, y_test = train_model(df)
    MODEL["model"] = model
    MODEL["X_test"] = X_test
    MODEL["y_test"] = y_test

    return {"message": "Model trained"}

@app.get("/bias")
def bias():
    model = MODEL.get("model")
    X_test = MODEL.get("X_test")
    y_test = MODEL.get("y_test")

    if model is None:
        return {"error": "Model not trained"}

    return calculate_bias(model, X_test, y_test)

@app.post("/mitigate")
def mitigate():
    df = DATA.get("df")
    if df is None:
        return {"error": "No dataset"}

    return mitigate_bias(df)

@app.get("/explain")
def explain():
    model = MODEL.get("model")
    if model is None:
        return {"error": "Model not trained"}

    return get_feature_importance(model)