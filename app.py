import streamlit as st
import pickle
import re

# Load pipeline
with open("sentiment_pipeline.pkl", "rb") as f:
    sentiment_pipeline = pickle.load(f)

# Clean text (basic)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())
    return text

# Streamlit UI
st.set_page_config(page_title="Social Media Sentiment Analysis")
st.title("📊 Social Media Sentiment Analysis")

input_text = st.text_area("Enter text to analyze:")

if st.button("Analyze Sentiment"):
    if input_text.strip() == "":
        st.warning("⚠ Please enter some text!")
    else:
        cleaned = preprocess_text(input_text)
        
        # Predict
        pred = sentiment_pipeline.predict([cleaned])[0]
        
        # Show result
        if pred == 1:
            st.success("😊 Positive sentiment")
        else:
            st.error("☹️ Negative sentiment")
