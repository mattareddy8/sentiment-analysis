# Sentiment Analysis using Naïve Bayes

This repository contains a **Sentiment Analysis** project using the **Naïve Bayes algorithm**.  
The model classifies text into **Positive, Negative, and Neutral** sentiments.

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/mattareddy8/sentiment-analysis-naive-bayes.git
cd sentiment-analysis-naive-bayes
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training Script
```bash
python sentiment_analysis.py
```

### 4. Run Web App
```bash
streamlit run app.py
```

## 📦 Project Structure
```
├── sentiment_analysis.py     # Main model training script
├── app.py                    # Streamlit app for predictions
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
```

## 🎯 Features
- Train Naïve Bayes classifier
- TF-IDF vectorization
- Model persistence with joblib
- Streamlit web app for real-time predictions

## 🌍 Deployment Options
- **Streamlit Cloud**
- **Hugging Face Spaces**
