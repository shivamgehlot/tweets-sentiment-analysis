import streamlit as st
from utils.preprocess import clean_text
from utils.predict import predict_tfidf, predict_lstm, predict_bert

st.set_page_config(page_title="Sentiment Analysis App", layout = "centered")

st.title("Sentiment Analysis - Model Comparison")

text = st.text_area("Enter text")

model_choice = st.selectbox(
    "Choose Model",
    [
        "TF-IDF + Logistic Regresssion",
        "LSTM",
        "BERT (Fine-Tuned)"
    ]
)

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter text.")
    else:
        cleaned_text = clean_text(text)

        if model_choice == "TF-IDF + Logistic Regresssion":
            prob = predict_tfidf(cleaned_text)

        elif model_choice == "LSTM":
            prob = predict_lstm(cleaned_text)

        else:
            prob = predict_bert(cleaned_text)

        sentiment = "Positive" if prob >= 0.5 else "Negative"

        st.subheader("Result : ")
        st.write(sentiment)
        st.progress(min(int(prob*100),100))
        st.write(f"Confidence: **{prob:.2f}**")
