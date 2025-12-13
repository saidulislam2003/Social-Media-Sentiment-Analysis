import streamlit as st
import joblib
import re
import numpy as np

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Sentiment Analysis Expert",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Sentiment Analysis App")
st.markdown(
    "Analyze text sentiment using **TF-IDF + Linear SVM** with expert threshold handling."
)

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Load Pipeline
# -----------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("sentiment_pipeline.pkl")

pipeline = load_pipeline()
vectorizer = pipeline.named_steps["tfidf"]
model = pipeline.named_steps["svm"]
classes = model.classes_

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "Enter text for sentiment analysis:",
    height=120,
    placeholder="Example: I really hate this product..."
)

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Analyze Sentiment"):

    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)

        # Vectorize
        X_vec = vectorizer.transform([cleaned_text])

        # Get decision function scores
        scores = model.decision_function(X_vec)[0]

        # Map scores to classes
        score_dict = dict(zip(classes, scores))

        # -----------------------------
        # Thresholding logic for all 3 classes
        # -----------------------------
        # Expert thresholds (can tweak after testing)
        negative_score = score_dict.get("negative", -1)
        neutral_score  = score_dict.get("neutral", -1)
        positive_score = score_dict.get("positive", -1)

        # Determine prediction
        if negative_score >= max(neutral_score, positive_score):
            prediction = "Negative 😠"
        elif positive_score >= max(neutral_score, negative_score):
            prediction = "Positive 😊"
        else:
            prediction = "Neutral 😐"

        # -----------------------------
        # Confidence Calculation
        # -----------------------------
        total_score = sum(np.abs([negative_score, neutral_score, positive_score]))
        confidence = round((score_dict[prediction.split()[0].lower()] / total_score) * 100, 2)

        # -----------------------------
        # Display Output
        # -----------------------------
        st.subheader("📊 Sentiment Prediction")
        st.success(f"{prediction} (Confidence: {confidence}%)")

        # -----------------------------
        # Detailed Scores for Debug / Expert View
        # -----------------------------
        with st.expander("🔍 Detailed Class Scores"):
            st.write(score_dict)
            st.write("Vectorized features shape:", X_vec.shape)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Expert Classical ML Deployment | TF-IDF + Linear SVM | Designed for accurate multi-class sentiment detection"
)
