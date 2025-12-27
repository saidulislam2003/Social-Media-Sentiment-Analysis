import streamlit as st
import joblib
import re

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Expert Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Expert Sentiment Analysis")
st.write("Accurate classical ML sentiment detection for positive, neutral, and negative inputs.")

# -----------------------------
# Load Pipeline
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("sentiment_pipeline.pkl")

pipeline = load_model()

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "Enter your text here:",
    placeholder="I really hate this...",
    height=120
)

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Clean input
        cleaned_text = clean_text(user_input)

        # Predict probabilities (IMPORTANT)
        proba = pipeline.predict_proba([cleaned_text])[0]
        classes = pipeline.classes_

        # Map class -> probability
        prob_dict = {
            label: float(prob)
            for label, prob in zip(classes, proba)
        }

        # Expert thresholds (tunable)
        NEGATIVE_THRESHOLD = 0.35
        POSITIVE_THRESHOLD = 0.35

        # Decision logic
        if prob_dict.get("negative", 0.0) >= NEGATIVE_THRESHOLD:
            prediction = "Negative 😠"
        elif prob_dict.get("positive", 0.0) >= POSITIVE_THRESHOLD:
            prediction = "Positive 😊"
        else:
            prediction = "Neutral 😐"

        # -----------------------------
        # Output
        # -----------------------------
        st.subheader("Prediction")
        st.success(prediction)

        # Probability display
        with st.expander("🔍 Probabilities"):
            for label, prob in prob_dict.items():
                st.write(f"**{label.capitalize()}**: {prob:.3f}")
