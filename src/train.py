"""
train.py
Train, evaluate, and save the best spam detection model.

Usage:
    python src/train.py --data data/spam.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
)

from preprocess import preprocess_text


# ── Configuration ────────────────────────────────────────────────────────────
MODELS = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(
        C=5, max_iter=1000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", max_iter=2000),
}

TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 2,
}

MODEL_DIR = "models"


def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate the dataset."""
    if filepath.endswith(".tsv"):
        df = pd.read_csv(filepath, sep="\t", header=None, names=["label", "message"])
    else:
        df = pd.read_csv(filepath, encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "message"]

    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    print(f"Loaded {len(df)} rows  |  Spam: {df['label_num'].mean()*100:.1f}%")
    return df


def train(data_path: str):
    """Full training pipeline."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load & preprocess
    df = load_data(data_path)
    df["clean"] = df["message"].apply(preprocess_text)

    X = df["clean"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

    # Train & evaluate
    best_auc = 0
    best_name = None
    best_clf = None

    print("\n" + "=" * 55)
    for name, clf in MODELS.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)

        try:
            scores = clf.predict_proba(X_test_vec)[:, 1]
        except AttributeError:
            raw = clf.decision_function(X_test_vec)
            scores = 1 / (1 + np.exp(-raw))

        auc = roc_auc_score(y_test, scores)
        print(f"{name:<25}  Acc={acc:.4f}  AUC={auc:.4f}")
        print(classification_report(y_test, preds, target_names=["ham", "spam"]))

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_clf = clf

    # Save best model + vectorizer
    print("=" * 55)
    print(f"\n🏆 Best model: {best_name}  (AUC={best_auc:.4f})")
    joblib.dump(best_clf, os.path.join(MODEL_DIR, "spam_model.pkl"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"✅ Saved to {MODEL_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spam detection model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/spam.csv",
        help="Path to the dataset CSV/TSV file",
    )
    args = parser.parse_args()
    train(args.data)
