import streamlit as st
import joblib
import string
import pandas as pd

# Load saved model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load dataset
try:
    data = pd.read_csv("fake_or_real_news.csv")
except FileNotFoundError:
    st.error("❌ Dataset file 'fake_or_real_news.csv' not found in app directory.")
    st.stop()

# Stopwords list (short version)
stop_words = set([
    "a", "an", "the", "and", "or", "is", "are", "was", "were", "be", "has", "have", "had",
    "of", "in", "to", "on", "for", "with", "as", "by", "at", "from", "that", "this"
])

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Paste your own news content or test random examples from the dataset.")

# User Input
news_text = st.text_area("✏️ Paste your news article here:", height=200)

if st.button("Check"):
    if not news_text.strip():
        st.warning("⚠️ Please enter some news content.")
    else:
        cleaned = preprocess(news_text)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        confidence = round(max(model.predict_proba(transformed)[0]) * 100, 2)
        result = "✅ Real News" if prediction == 1 else "🚫 Fake News"
        st.success(f"**Prediction:** {result}  \n**Confidence:** {confidence}%")

# Divider
st.divider()

# Test random dataset sample
st.subheader("🎲 Try a Random News Article from Dataset")

if st.button("Show Example"):
    sample = data.sample(1).iloc[0]
    st.write("**Article Text:**")
    st.write(sample['text'])

    cleaned = preprocess(sample['text'])
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)[0]
    confidence = round(max(model.predict_proba(transformed)[0]) * 100, 2)
    result = "✅ Real News" if prediction == 1 else "🚫 Fake News"

    st.info(f"**Actual Label:** {sample['label']}")
    st.success(f"**Prediction:** {result}  \n**Confidence:** {confidence}%")
