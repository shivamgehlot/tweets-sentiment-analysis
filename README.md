# 🧠 Sentiment Analysis Web App

**TF-IDF | LSTM | BERT (Fine-Tuned)**

An end-to-end **Sentiment Analysis system** that compares three different NLP approaches — **Traditional ML**, **Deep Learning**, and **Transformer-based models** — deployed using **Streamlit**.

---

## 🚀 Live Demo

🔗 **

---

## 📌 Project Overview

This project performs **binary sentiment classification (Positive / Negative)** on text data using:

1. **TF-IDF + Logistic Regression**
2. **LSTM Neural Network**
3. **Fine-tuned BERT (bert-base-uncased)**

The goal is to **compare accuracy, performance, and inference behavior** across different NLP paradigms.

---

## 🏗️ Architecture

```
User Input
   │
   ├── TF-IDF + Logistic Regression
   ├── LSTM (Keras)
   └── BERT (Transformers)
        │
        ▼
   Sentiment Prediction + Confidence
```

---

## 🧪 Models Used

### 1️⃣ TF-IDF + Logistic Regression

* Feature extraction using **TF-IDF**
* Classifier: **Logistic Regression**
* Fast inference, lightweight

### 2️⃣ LSTM (Deep Learning)

* Tokenization + padding (`max_len = 360`)
* Embedding + LSTM layers
* Better contextual understanding than TF-IDF

### 3️⃣ BERT (Transformer)

* **bert-base-uncased**
* Fine-tuned using Hugging Face `Trainer`
* Highest accuracy, context-aware

---

## 📊 Training Results (Example)

| Model  | Validation Accuracy |
| ------ | ------------------- |
| TF-IDF | ~83%                |
| LSTM   | ~84%                |
| BERT   | ~86%                |

---

## 🖥️ Web Application (Streamlit)

Features:

* Single text input
* Real-time predictions from **all three models**
* Confidence score display
* Clean UI for comparison

---

## 📁 Project Structure

```
tweets-sentiment-analysis/
│
├── app.py                  # Streamlit app
├── utils/
│   ├── preprocess.py       # Text cleaning
│   └── predict.py          # Model inference
│
├── notebook/               # Training notebooks
├── models/                 # (ignored in GitHub)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚠️ Important Note About Models

> **Trained models are NOT included in this repository**

Reason:

* Large file sizes
* GitHub limitations
* Best practices

### Models are loaded from:

* Local storage (development)
* Hugging Face Hub (deployment)

---

## 🛠️ Installation & Setup

### 1️⃣ Clone repository

```bash
git clone https://github.com/shivamgehlot/tweets-sentiment-analysis.git
cd tweets-sentiment-analysis
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit app

```bash
streamlit run app.py
```

---

## 📦 Dependencies

* Python 3.9+
* streamlit
* scikit-learn
* tensorflow
* torch
* transformers
* safetensors
* joblib
* numpy
* pandas
* huggingface_hub

---

## 🎯 Key Learnings

* Difference between traditional ML, DL, and transformer models
* Model deployment considerations
* Managing large ML artifacts
* Streamlit deployment workflow
* Hugging Face Transformers usage

---

## 🧑‍💻 Author

**Shivam Gehlot**
Software Engineering | Machine Learning | NLP

🔗 GitHub: [https://github.com/shivamgehlot](https://github.com/shivamgehlot)

---

## ⭐ Future Improvements

* Optimize BERT inference
* Add multilingual support

---

## 📜 License

This project is for **educational and portfolio purposes**.


