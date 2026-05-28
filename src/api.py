"""
api.py
FastAPI REST API for spam detection.

Usage:
    uvicorn src.api:app --reload
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

from preprocess import preprocess_text

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Spam Detection API",
    description="Classify messages as spam or ham using a trained ML model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"

# ── Load model at startup ────────────────────────────────────────────────────
clf = None
vectorizer = None


@app.on_event("startup")
def load_model():
    global clf, vectorizer
    clf_path = os.path.join(MODEL_DIR, "spam_model.pkl")
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(clf_path):
        print(f"⚠️  Model not found at {clf_path}. Run train.py first.")
        return
    clf = joblib.load(clf_path)
    vectorizer = joblib.load(vec_path)
    print("✅ Model loaded successfully")


# ── Schemas ───────────────────────────────────────────────────────────────────
class MessageRequest(BaseModel):
    text: str = Field(..., min_length=1, example="Congratulations! You won a FREE prize!")


class MessageResponse(BaseModel):
    text: str
    label: str
    confidence: float
    is_spam: bool


class BatchRequest(BaseModel):
    messages: List[str] = Field(..., min_items=1, max_items=100)


class BatchResponse(BaseModel):
    results: List[MessageResponse]
    total: int
    spam_count: int


# ── Helpers ───────────────────────────────────────────────────────────────────
def _predict(text: str) -> MessageResponse:
    if clf is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python src/train.py` first.",
        )
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    pred = int(clf.predict(vec)[0])
    try:
        prob = float(clf.predict_proba(vec)[0][1])
    except AttributeError:
        raw = float(clf.decision_function(vec)[0])
        prob = 1 / (1 + np.exp(-raw))

    confidence = prob if pred == 1 else 1 - prob
    return MessageResponse(
        text=text,
        label="spam" if pred else "ham",
        confidence=round(confidence, 4),
        is_spam=bool(pred),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "message": "Spam Detection API is running 🚀"}


@app.post("/predict", response_model=MessageResponse, summary="Predict single message")
def predict(request: MessageRequest):
    """
    Classify a single message as **spam** or **ham**.

    Returns the label and confidence score (0–1).
    """
    return _predict(request.text)


@app.post("/predict/batch", response_model=BatchResponse, summary="Predict batch of messages")
def predict_batch(request: BatchRequest):
    """
    Classify up to **100 messages** in a single request.
    """
    results = [_predict(msg) for msg in request.messages]
    spam_count = sum(r.is_spam for r in results)
    return BatchResponse(results=results, total=len(results), spam_count=spam_count)
