"""
predict.py
Load a saved model and predict spam / ham for a given message.

Usage:
    python src/predict.py --message "You won a FREE prize!"
    python src/predict.py --file messages.txt
"""

import argparse
import os
import numpy as np
import joblib

from preprocess import preprocess_text

MODEL_DIR = "models"


def load_model():
    """Load saved model and vectorizer from disk."""
    clf_path = os.path.join(MODEL_DIR, "spam_model.pkl")
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    if not os.path.exists(clf_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(
            f"Model files not found in '{MODEL_DIR}/'. "
            "Run `python src/train.py` first."
        )

    clf = joblib.load(clf_path)
    vectorizer = joblib.load(vec_path)
    return clf, vectorizer


def predict_spam(message: str, clf=None, vectorizer=None) -> dict:
    """
    Predict whether a single message is spam or ham.

    Args:
        message:    Raw input text.
        clf:        Trained classifier (loaded automatically if None).
        vectorizer: TF-IDF vectorizer (loaded automatically if None).

    Returns:
        dict with keys: message, label, confidence
    """
    if clf is None or vectorizer is None:
        clf, vectorizer = load_model()

    clean = preprocess_text(message)
    vec = vectorizer.transform([clean])
    pred = clf.predict(vec)[0]

    try:
        prob = clf.predict_proba(vec)[0][1]
    except AttributeError:
        raw = clf.decision_function(vec)[0]
        prob = 1 / (1 + np.exp(-raw))

    confidence = prob if pred == 1 else 1 - prob

    return {
        "message": message,
        "label": "spam" if pred == 1 else "ham",
        "confidence": round(float(confidence), 4),
    }


def predict_batch(messages: list, clf=None, vectorizer=None) -> list:
    """Predict spam/ham for a list of messages."""
    if clf is None or vectorizer is None:
        clf, vectorizer = load_model()
    return [predict_spam(m, clf, vectorizer) for m in messages]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spam prediction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--message", type=str, help="Single message to classify")
    group.add_argument("--file", type=str, help="Text file with one message per line")
    args = parser.parse_args()

    clf, vectorizer = load_model()

    if args.message:
        result = predict_spam(args.message, clf, vectorizer)
        print(f"\nMessage    : {result['message']}")
        print(f"Label      : {result['label'].upper()}")
        print(f"Confidence : {result['confidence']*100:.1f}%")

    elif args.file:
        with open(args.file) as f:
            messages = [line.strip() for line in f if line.strip()]

        results = predict_batch(messages, clf, vectorizer)
        print(f"\n{'Message':<55} {'Label':<8} Confidence")
        print("-" * 75)
        for r in results:
            short = (r["message"][:52] + "...") if len(r["message"]) > 52 else r["message"]
            print(f"{short:<55} {r['label']:<8} {r['confidence']*100:.1f}%")
