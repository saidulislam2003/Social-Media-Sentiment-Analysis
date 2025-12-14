import streamlit as st
import joblib
import re

st.set_page_config(
    page_title="Expert Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Expert Sentiment Analysis")
st.write("Accurate classical ML sentiment detection for positive, neutral, and negative inputs.")

@st.cache_resource
def load_model():
    return joblib.load("sentiment_pipeline.pkl")

pipeline = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


user_input = st.text_area(
    "Enter your text here:",
    placeholder="I really hate this...",
    height=120
)


if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        X_vec = [cleaned_text]


        proba = pipeline.predict_proba(X_vec)[0]
        classes = pipeline.classes_

        prob_dict = dict(zip(classes, proba))

        NEGATIVE_THRESHOLD = 0.35
        POSITIVE_THRESHOLD = 0.35

        if prob_dict.get("negative", 0) >= NEGATIVE_THRESHOLD:
            prediction = "Negative 😠"
        elif prob_dict.get("positive", 0) >= POSITIVE_THRESHOLD:
            prediction = "Positive 😊"
        else:
            prediction = "Neutral 😐"

        st.subheader("Prediction:")
        st.success(prediction)

        with st.expander("🔍 Probabilities"):
            st.write(prob_dict)

