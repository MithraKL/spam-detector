# 📧 Spam Message Detection System

An end-to-end Machine Learning pipeline to classify SMS/email messages as **spam** or **ham** (not spam), with multiple model comparisons, TF-IDF vectorization, evaluation metrics, and a FastAPI deployment endpoint.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Features

- ✅ Full NLP preprocessing (lowercasing, stemming, stopword removal, URL/phone normalization)
- ✅ TF-IDF vectorization with unigrams + bigrams
- ✅ 4 models compared: Naive Bayes, Logistic Regression, Random Forest, Linear SVM
- ✅ Evaluation: Accuracy, F1, ROC-AUC, Confusion Matrix, ROC Curves
- ✅ Feature importance visualization
- ✅ 5-fold cross-validation
- ✅ FastAPI REST endpoint for real-time prediction
- ✅ Google Colab-ready Jupyter Notebook

---

## 📁 Project Structure

```
spam-detector/
│
├── spam_detection.ipynb     # Main Jupyter / Colab notebook
├── src/
│   ├── preprocess.py        # Text cleaning & preprocessing
│   ├── train.py             # Model training & saving
│   ├── predict.py           # Prediction function
│   └── api.py               # FastAPI REST API
│
├── models/                  # Saved model files (auto-created on train)
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── data/
│   └── README.md            # Dataset download instructions
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/spam-detector.git
cd spam-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get the dataset
Download [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle and place `spam.csv` in the `data/` folder.

### 5. Train the model
```bash
python src/train.py
```

### 6. Run the API
```bash
uvicorn src.api:app --reload
```

Then test it:
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Congratulations! You won a FREE iPhone. Click now!"}'
```

---

## 🌐 API Endpoints

| Method | Endpoint    | Description              |
|--------|-------------|--------------------------|
| GET    | `/`         | Health check             |
| POST   | `/predict`  | Predict spam or ham      |

**Request body:**
```json
{ "text": "Your message here" }
```

**Response:**
```json
{ "label": "spam", "confidence": 0.9871 }
```

---

## 📊 Model Performance (on SMS Spam Collection)

| Model               | Accuracy | AUC-ROC |
|---------------------|----------|---------|
| Naive Bayes         | ~97.4%   | ~0.985  |
| Logistic Regression | ~98.1%   | ~0.992  |
| Random Forest       | ~97.8%   | ~0.991  |
| Linear SVM          | ~98.3%   | ~0.994  |

---

## ☁️ Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/spam-detector/blob/main/spam_detection.ipynb)

> Replace `YOUR_USERNAME` with your GitHub username after pushing.

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **ML/NLP:** scikit-learn, NLTK
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **API:** FastAPI, Uvicorn
- **Serialization:** Joblib

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
