import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load saved vectorizer and classifier
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Text cleaning function (same preprocessing used during training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    text = " ".join(tokens)
    return text

# Streamlit app layout
st.set_page_config(page_title="Social Media Sentiment Analysis")
st.title("📊 Social Media Sentiment Analyzer")

input_text = st.text_area("Enter text to analyze:")

if st.button("Analyze Sentiment"):
    if input_text.strip() == "":
        st.error("⚠ Please enter some text to analyze!")
    else:
        # Preprocess user text
        cleaned = preprocess_text(input_text)
        
        # Vectorize and predict
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        
        # Output result
        if pred == 1:
            st.success("😊 Positive sentiment")
        else:
            st.error("☹️ Negative sentiment")
