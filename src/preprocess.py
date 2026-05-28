"""
preprocess.py
Text cleaning and NLP preprocessing pipeline for spam detection.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stemmer = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline.

    Steps:
        1. Lowercase
        2. Replace URLs with 'url'
        3. Replace email addresses with 'email'
        4. Replace phone numbers with 'phone'
        5. Remove punctuation & special characters
        6. Tokenise
        7. Remove stopwords & apply stemming

    Args:
        text: Raw input message string.

    Returns:
        Cleaned, stemmed string ready for vectorization.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "url", text)
    text = re.sub(r"\S+@\S+", "email", text)
    text = re.sub(r"\b\d[\d\s\-\.]{7,}\d\b", "phone", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [
        stemmer.stem(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 1
    ]
    return " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "WINNER!! You won a FREE iPhone! Click http://scam.com now!",
        "Hey, are you free tomorrow for lunch?",
        "Call 0800-123-4567 to claim your prize!",
    ]
    for s in samples:
        print(f"Original : {s}")
        print(f"Cleaned  : {preprocess_text(s)}\n")
