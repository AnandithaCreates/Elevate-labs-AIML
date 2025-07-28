# 📰 Fake News Classification using NLP & Logistic Regression

## 🚀 Project Overview
A robust NLP pipeline for classifying news articles as *Fake* or *Real* using Logistic Regression and TF-IDF vectorization. This project demonstrates end-to-end development—from data ingestion to deployment in a Streamlit app—showcasing expertise in supervised learning, feature engineering, and model explainability.

---

## 🧠 Problem Statement
Fake news poses a serious threat in digital media, distorting facts and public perception. This project addresses the challenge by detecting misinformation through statistical analysis of linguistic features.

---

## 🗂️ Dataset
Source: [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

- **Fake.csv:** ~23,000 articles labeled as fake  
- **True.csv:** ~21,000 articles labeled as real  
- Combined → 45,000+ articles in a balanced, shuffled dataset

---

## 🛠️ Tools & Libraries

| Tool/Libraries       | Purpose                          |
|----------------------|----------------------------------|
| Python               | Core programming language        |
| Pandas               | Data loading & manipulation      |
| NLTK                 | Text cleaning, stopword removal  |
| Scikit-learn         | TF-IDF, model training & metrics |
| Matplotlib & Seaborn | Visualizations                   |
| Pickle               | Model & vectorizer serialization |
| Streamlit            | Web app deployment               |

---

## 🔄 Project Workflow

### 1. Data Preparation
- Merge `Fake.csv` and `True.csv`
- Assign labels: `0 = Fake`, `1 = Real`
- Combine `title` + `text` → `full_text`

### 2. Text Preprocessing (NLTK)
- Lowercase conversion  
- Remove URLs, punctuation, numbers, stopwords  
- Output stored in `cleaned_text`

### 3. Feature Engineering
- TF-IDF with:
  - `max_features = 10,000`
  - `min_df = 5`  
- Output: Sparse matrix `X_tfidf`

### 4. Model Training
- Train-test split (80:20) with stratification  
- Logistic Regression with `max_iter = 1000`

### 5. Evaluation Metrics
- **Accuracy:** 98.89%  
- **Precision, Recall, F1-score:** ~0.99  
- **Confusion Matrix:** High true positives, minimal errors

### 6. Model Persistence
- Save model & vectorizer via Pickle:
  - `logistic_regression_model.pkl`
  - `tfidf_vectorizer.pkl`

### 7. Deployment
- Streamlit app: `fake_news_app.py`
- Inputs article text, predicts label + confidence  
- Interactive explanation interface

---

## 📊 Visual Output
- `logistic_regression_confusion_matrix.png`  
  Visual summary of model performance

---

## 🔍 Key Highlights
- End-to-end reproducible ML workflow  
- Scalable TF-IDF feature extraction  
- Real-time prediction via Streamlit  
- High interpretability and deployment readiness

---

## 🎯 Outcome
Delivered a production-grade, fake news classification system with strong performance metrics and user-friendly deployment. Ideal for real-world scenarios in media monitoring, content filtering, and platform trust management.

---
