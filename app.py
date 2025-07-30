# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 18:22:32 2025
@author: Ananditha R
"""

import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

# --- Streamlit Page Setup ---
st.set_page_config(page_title="News Authenticity Checker", page_icon="🔍")

# --- Load Model & Vectorizer Safely ---
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open("logistic_regression_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Required model/vectorizer files are missing.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

model, tfidf_vectorizer = load_model_and_vectorizer()

# --- Clean Text Function ---
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# --- App Interface ---
st.title("🔍 News Authenticity Checker")
st.markdown("Use this tool to verify whether a news article is **Fake** or **Real** based on its content.")

user_input = st.text_area("📰 Paste your news article content here:", height=250)

if st.button("Check Authenticity"):
    if user_input.strip():
        cleaned = preprocess_text(user_input)
        transformed = tfidf_vectorizer.transform([cleaned])
        prediction = model.predict(transformed)
        probability = model.predict_proba(transformed)

        st.subheader("🧾 Prediction Result:")
        if prediction[0] == 0:
            st.error("🚩 This article seems **FAKE**.")
            st.markdown(f"Confidence Score: **{probability[0][0]*100:.2f}% Fake**, **{probability[0][1]*100:.2f}% Real**")
        else:
            st.success("✅ This article appears **REAL**.")
            st.markdown(f"Confidence Score: **{probability[0][1]*100:.2f}% Real**, **{probability[0][0]*100:.2f}% Fake**")

        st.markdown("---")
        st.info("This classification is based on a Logistic Regression model using TF-IDF word analysis. While it's trained on a large dataset, always verify facts from reliable sources.")
    else:
        st.warning("Please enter some article content to analyze.")

st.markdown("---")
st.caption("👩‍💻 Developed by Ananditha R • AIML Internship Project • 2025")
