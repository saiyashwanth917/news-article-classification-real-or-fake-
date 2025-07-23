import streamlit as st
import joblib
import string

# Load saved model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Stopwords list (short version)
stop_words = set([
    "a", "an", "the", "and", "or", "is", "are", "was", "were", "be", "has", "have", "had",
    "of", "in", "to", "on", "for", "with", "as", "by", "at", "from", "that", "this"
])

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("📰 Fake News Detector")

news_text = st.text_area("Paste your news article here:")

if st.button("Check"):
    if not news_text.strip():
        st.warning("Please enter some news content.")
    else:
        cleaned = preprocess(news_text)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        result = "✅ Real News" if prediction == 1 else "🚫 Fake News"
        st.success(f"Prediction: {result}")
