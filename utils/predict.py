import torch
import joblib
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification

# laoding models
logreg = joblib.load("models/lg/logreg.pkl")
tfidf = joblib.load("models/lg/tfidf.pkl")

lstm_model = tf.keras.models.load_model("models/lstm/sentiment_lstm_model.keras")
lstm_tokenizer = joblib.load("models/lstm/tokenizer.pkl")

bert_tokenizer = BertTokenizer.from_pretrained("models/bert")
bert_model = BertForSequenceClassification.from_pretrained("models/bert")
bert_model.eval()



# prediction function
def predict_tfidf(text):
    vec = tfidf.transform([text])
    prob = logreg.predict_proba(vec)[0][1]
    return prob

def predict_lstm(text):
    seq = lstm_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=360, padding="pre")
    prob = lstm_model.predict(padded, verbose=0)[0][0]
    return float(prob)

def predict_bert(text):
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


