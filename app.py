import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📝 Sentiment Analysis App")
st.write("Classify text into Positive, Negative, or Neutral sentiments using Naïve Bayes.")

user_input = st.text_area("Enter text here:")

if st.button("Analyze"):
    if user_input.strip():
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)[0]
        st.success(f"Predicted Sentiment: {prediction}")
    else:
        st.warning("Please enter some text.")
