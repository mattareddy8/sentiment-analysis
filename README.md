# Sentiment Analysis

A text classification model that predicts sentiment (positive/negative) from real-world text data.

## Tech Stack
Python, Scikit-learn, NLTK, TF-IDF, Naïve Bayes

## How It Works
- Preprocesses raw text with tokenization and stop-word removal using NLTK
- Vectorizes text using TF-IDF
- Trains a Naïve Bayes classifier; achieved **87% accuracy** after hyperparameter tuning
- Validated with precision, recall, and F1 metrics across sentiment classes

## Run Locally
```bash
pip install scikit-learn nltk
python model.py
```
