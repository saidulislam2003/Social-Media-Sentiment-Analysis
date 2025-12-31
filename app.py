import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

# Load the saved vectorizer and model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Text preprocessing (cleaning)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# App UI
st.set_page_config(page_title="Social Media Sentiment Analysis")
st.title("📊 Social Media Sentiment Analysis")

input_text = st.text_area("Enter text to analyze:")

if st.button("Analyze Sentiment"):
    if not input_text.strip():
        st.warning("⚠ Please enter text first!")
    else:
        cleaned = preprocess_text(input_text)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]

        # Show result
        if prediction == 1:
            st.success("😊 Positive sentiment")
        else:
            st.error("☹️ Negative sentiment")
