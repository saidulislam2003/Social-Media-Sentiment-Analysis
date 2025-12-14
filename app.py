import streamlit as st
import joblib
import re
import os


st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Sentiment Analysis")
st.write("Classic ML-based sentiment analysis with expert corrections")


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "sentiment_pipeline_expert.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipeline = load_model()

vectorizer = pipeline.named_steps["tfidf"]
model = pipeline.named_steps["svm"]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


STRONG_NEGATIVE_WORDS = {
    "hate", "hated", "hating",
    "worst", "awful", "terrible",
    "disgusting", "horrible",
    "pathetic", "useless",
    "trash", "garbage",
    "bad", "poor", "annoying"
}

STRONG_POSITIVE_WORDS = {
    "love", "loved", "loving",
    "amazing", "excellent",
    "awesome", "fantastic",
    "perfect", "wonderful",
    "great"
}

user_input = st.text_area(
    "Enter your text:",
    placeholder="I really hate this product...",
    height=120
)


if st.button("Analyze Sentiment"):

    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        tokens = set(cleaned.split())

        if tokens & STRONG_NEGATIVE_WORDS:
            sentiment = "Negative 😠"
            reason = "Detected strong negative keywords"

        elif tokens & STRONG_POSITIVE_WORDS:
            sentiment = "Positive 😊"
            reason = "Detected strong positive keywords"

        else:
            X_vec = vectorizer.transform([cleaned])
            scores = model.decision_function(X_vec)[0]
            classes = model.classes_

            score_dict = dict(zip(classes, scores))

            if score_dict.get("negative", -1) > 0.2:
                sentiment = "Negative 😠"
            elif score_dict.get("positive", -1) > 0:
                sentiment = "Positive 😊"
            else:
                sentiment = "Neutral 😐"

            reason = "Predicted by ML model"


        st.subheader("Prediction")
        st.success(sentiment)

        with st.expander("ℹ️ Prediction Details"):
            st.write("Cleaned text:", cleaned)
            st.write("Reason:", reason)

            if "score_dict" in locals():
                st.write("Decision scores:", score_dict)
