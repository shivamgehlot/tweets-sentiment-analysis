import streamlit as st
import torch
import joblib
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download

# laoding models
@st.cache_resource
def load_tfidf():
    tfidf_path = hf_hub_download(
        repo_id = "shivamgehlot/logreg-sentiment",
        filename = "tfidf.pkl"
    )
    logreg_path = hf_hub_download(
        repo_id = "shivamgehlot/logreg-sentiment",
        filename = "logreg.pkl"
    )
    logreg = joblib.load(logreg_path)
    tfidf = joblib.load(tfidf_path)
    return tfidf, logreg

@st.cache_resource
def load_lstm():
    tokenizer_path = hf_hub_download(
        repo_id = "shivamgehlot/lstm-sentiment",
        filename = "tokenizer.pkl"
    )
    model_path = hf_hub_download(
        repo_id = "shivamgehlot/lstm-sentiment",
        filename = "sentiment_lstm_model.keras"
    )
    lstm_model = tf.keras.models.load_model(model_path)
    lstm_tokenizer = joblib.load(tokenizer_path)
    return lstm_tokenizer, lstm_model

@st.cache_resource
def load_bert():
    bert_tokenizer = BertTokenizer.from_pretrained("shivamgehlot/bert-sentiment")
    bert_model = BertForSequenceClassification.from_pretrained("shivamgehlot/bert-sentiment")
    bert_model.eval()
    return bert_tokenizer, bert_model



# prediction function
def predict_tfidf(text):
    tfidf, logreg = load_tfidf()
    vec = tfidf.transform([text])
    prob = logreg.predict_proba(vec)[0][1]
    return prob

def predict_lstm(text):
    lstm_tokenizer, lstm_model = load_lstm()
    seq = lstm_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=360, padding="pre")
    prob = lstm_model.predict(padded, verbose=0)[0][0]
    return float(prob)

def predict_bert(text):
    bert_tokenizer, bert_model = load_bert()
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    prob = torch.softmax(logits, dim=1)[0][1].item()
    return prob


