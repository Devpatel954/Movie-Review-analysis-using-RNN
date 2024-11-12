import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the IMDB word index
from tensorflow.keras.datasets import imdb
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained RNN model
model = load_model("rnn_p.h5")

# Set the maximum length for padding
max_len = 500

# Streamlit app title
st.title("🎬 IMDB Movie Review Sentiment Analysis 🎭")

# Function to decode encoded reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input text
def preprocess_text(text):
    words = text.lower().split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Encode words using the IMDB word index
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # Pad the sequence to the maximum length
    padded_review = pad_sequences([encoded_review], maxlen=max_len)
    return padded_review

# Function to predict sentiment of the review
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive 😊' if prediction[0][0] > 0.5 else 'Negative 😞'
    return sentiment, prediction[0][0]

# Streamlit app user input
st.header("Enter a Movie Review for Sentiment Analysis")
user_review = st.text_area("Type your review here...")

# Button to trigger prediction
if st.button("Analyze Sentiment"):
    if user_review:
        sentiment, score = predict_sentiment(user_review)
        st.subheader("Prediction Results")
        st.write(f"Sentiment: **{sentiment}**")
        st.write(f"Confidence Score: **{score:.2f}**")
    else:
        st.error("Please enter a review to analyze.")

# Optional: Display an example review and its prediction
if st.button("Show Example Review"):
    example_review = "This movie was thrilling and characters were performing extremely well"
    sentiment, score = predict_sentiment(example_review)
    st.write(f"Example Review: *{example_review}*")
    st.write(f"Sentiment: **{sentiment}**")
    st.write(f"Confidence Score: **{score:.2f}**")
