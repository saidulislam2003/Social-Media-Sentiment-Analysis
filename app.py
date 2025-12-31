import streamlit as st
import pickle
import re

# Load the saved sentiment pipeline
with open("models/sentiment_pipeline.pkl", "rb") as f:
    sentiment_pipeline = pickle.load(f)

# Basic text cleaning (no NLTK)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation and numbers
    text = " ".join(text.split())            # remove extra spaces
    return text

# Streamlit UI
st.set_page_config(page_title="Social Media Sentiment Analysis")
st.title("📊 Social Media Sentiment Analysis")

input_text = st.text_area("Enter text to analyze:")

if st.button("Analyze Sentiment"):
    if not input_text.strip():
        st.warning("⚠ Please enter some text!")
    else:
        # Clean and predict
        cleaned = preprocess_text(input_text)
        pred = sentiment_pipeline.predict([cleaned])[0]

        # Display
        if pred == 1:
            st.success("😊 Positive sentiment")
        else:
            st.error("☹️ Negative sentiment")

