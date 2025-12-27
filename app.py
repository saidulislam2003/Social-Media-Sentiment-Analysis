import streamlit as st
import joblib
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# -------------------------------
# Constants
# -------------------------------
MODEL_PATH = "models/sentiment_pipeline.pkl"

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    """
    Load the trained sentiment analysis pipeline.
    Cached to avoid reloading on every interaction.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Model file not found at: {MODEL_PATH}\n"
            f"Make sure the file exists and is pushed to GitHub."
        )
    return joblib.load(MODEL_PATH)

# Load model safely
try:
    pipeline = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

# -------------------------------
# UI
# -------------------------------
st.title("💬 Social Media Sentiment Analysis")
st.write("Analyze sentiment of social media text using a trained ML model.")

user_input = st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Type a tweet, comment, or post..."
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            prediction = pipeline.predict([user_input])[0]

            # Optional: probability support
            if hasattr(pipeline, "predict_proba"):
                prob = pipeline.predict_proba([user_input]).max()

            # Display result
            st.subheader("Result")

            if prediction.lower() == "positive":
                st.success(f"😊 Positive sentiment")
            elif prediction.lower() == "negative":
                st.error(f"😠 Negative sentiment")
            else:
                st.info(f"😐 Neutral sentiment")

            if hasattr(pipeline, "predict_proba"):
                st.caption(f"Confidence: {prob:.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------
# Debug Section (Remove later)
# -------------------------------
with st.expander("🔍 Debug Info"):
    st.write("Current directory:", os.getcwd())
    st.write("Files in root:", os.listdir())
    if os.path.exists("models"):
        st.write("Files in models/:", os.listdir("models"))
