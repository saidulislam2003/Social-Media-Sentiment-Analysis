import streamlit as st
import joblib
import re
import os

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Expert Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Sentiment Analysis")
st.write("Classic ML-based sentiment analyzer")

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "sentiment_pipeline.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipeline = load_model()

# Extract components
vectorizer = pipeline.named_steps["tfidf"]
model = pipeline.named_steps["svm"]

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Input
# -----------------------------
user_input = st.text_area(
    "Enter your text:",
    placeholder="I really hate this product...",
    height=120
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Analyze Sentiment"):

    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)

        X_vec = vectorizer.transform([cleaned])

        # Decision scores
        scores = model.decision_function(X_vec)[0]
        classes = model.classes_

        score_dict = dict(zip(classes, scores))

        # -----------------------------
        # Expert Threshold Logic
        # -----------------------------
        if score_dict.get("negative", -1) > 0.5:
            sentiment = "Negative 😠"
        elif score_dict.get("positive", -1) > 0:
            sentiment = "Positive 😊"
        else:
            sentiment = "Neutral 😐"

        # -----------------------------
        # Output
        # -----------------------------
        st.subheader("Prediction")
        st.success(sentiment)

        with st.expander("🔍 Decision Scores (Debug)"):
            st.write(score_dict)
