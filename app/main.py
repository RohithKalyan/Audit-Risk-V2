from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import requests
import uuid
import os
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from catboost import CatBoostClassifier
import shap
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Lazy load models ===
catboost_model = None
model_bert = None

# === Input model ===
class PredictionRequest(BaseModel):
    file_url: str
    callback_url: Optional[str] = None

# === Prediction Logic ===
def run_prediction_pipeline(file_url):
    global catboost_model, model_bert

    if catboost_model is None:
        catboost_model = CatBoostClassifier()
        catboost_model.load_model("models/catboost_v2_model.cbm")

    if model_bert is None:
        model_bert = SentenceTransformer("all-MiniLM-L6-v2")

    df = pd.read_csv(file_url)
    df.columns = df.columns.str.strip()

    # Clean numerics
    comma_cols = ["Entered Dr SUM", "Entered Cr SUM", "Accounted Dr SUM", "Accounted Cr SUM", "Net Amount"]
    for col in comma_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").replace("nan", np.nan).astype(float)

    # Combine text
    text_fields = ["Line Desc", "Source Desc", "Batch Name"]
    df[text_fields] = df[text_fields].fillna("")
    df["Combined_Text"] = df["Line Desc"] + " | " + df["Source Desc"] + " | " + df["Batch Name"]

    # BERT Embeddings + Clustering
    embeddings = model_bert.encode(df["Combined_Text"].tolist(), show_progress_bar=False)
    embedding_df = pd.DataFrame(embeddings, columns=[f"text_emb_{i}" for i in range(embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)

    umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reduced = umap_model.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=10, random_state=42)
    df["Narration_Cluster"] = kmeans.fit_predict(reduced)

    cluster_summary = (
        df.groupby("Narration_Cluster")["Combined_Text"]
        .apply(lambda x: "; ".join(x.head(3)))
        .reset_index(name="Narration_Cluster_Label")
    )
    df = df.merge(cluster_summary, on="Narration_Cluster", how="left")

    # Date features
    for col in ["Accounting Date", "Invoice Date", "Posted Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["Accounting_Month"] = df["Accounting Date"].dt.month
    df["Accounting_Weekday"] = df["Accounting Date"].dt.weekday
    df["Invoice_Month"] = df["Invoice Date"].dt.month
    df["Invoice_Weekday"] = df["Invoice Date"].dt.weekday
    df["Posted_Month"] = df["Posted Date"].dt.month
    df["Posted_Weekday"] = df["Posted Date"].dt.weekday

    # Features
    exclude_cols = ["S. No", "Combined_Text", "Accounting Date", "Invoice Date", "Posted Date"]
    model_feature_names = catboost_model.feature_names_
    feature_cols = [col for col in df.columns if col in model_feature_names and col not in exclude_cols and not col.startswith("Unnamed")]

    for col in feature_cols:
        if df[col].dtype == object or df[col].isnull().any():
            df[col] = df[col].astype(str).fillna("Missing")

    X_final = df[feature_cols].copy()

    # Prediction
    df["Model_Score"] = catboost_model.predict_proba(X_final)[:, 1]
    df["Final_Score"] = df["Model_Score"].round(3)

    # Placeholder Explanation Fields (replace with SHAP + CP logic if needed)
    df["Top_Risky_Feature_Groups"] = "- Placeholder Risky Feature"
    df["Top_Safe_Feature_Groups"] = "- Placeholder Safe Feature"
    df["Explanation_Summary"] = "⚠️ Placeholder Summary"
    df["Triggered_CPs"] = "CP_01 (83), CP_02 (86)"
    df["CP_Score"] = 0.95

    columns = ["S. No", "Model_Score", "Top_Risky_Feature_Groups", "Top_Safe_Feature_Groups", "Explanation_Summary", "Triggered_CPs", "CP_Score"]
    results = df[columns].fillna("").to_dict(orient="records")
    return results

# === Async Predict Endpoint ===
@app.post("/predict_async")
def predict_async(req: PredictionRequest):
    try:
        results = run_prediction_pipeline(req.file_url)
        if req.callback_url:
            requests.post(req.callback_url, json=results)
        return {"detail": "Prediction started. Results will be sent to callback_url."}
    except Exception as e:
        return {"error": str(e)}
