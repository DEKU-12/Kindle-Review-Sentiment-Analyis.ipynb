# Kindle Review Sentiment Analysis (Ratings Classification)

This project analyzes Kindle reviews and predicts the review **rating** using NLP + machine learning.

## Problem
Online reviews contain valuable feedback, but reading them manually doesn’t scale.  
This project converts review text into numerical features and trains classification models to predict the review rating.

## What I built
### 1) Text preprocessing
- Removed special characters and extra spaces
- Removed stopwords (NLTK)
- Removed URLs and HTML tags (BeautifulSoup)
- Lemmatized words using WordNet Lemmatizer

### 2) Feature extraction (3 approaches)
- **Bag of Words (BoW)** using `CountVectorizer`
- **TF-IDF** using `TfidfVectorizer`
- **Word2Vec embeddings** using pretrained **Google News 300D** vectors (gensim), with an **average word vector per review**

### 3) Modeling + evaluation
- Trained **Gaussian Naive Bayes** classifiers on:
  - BoW features
  - TF-IDF features
  - Word2Vec averaged embeddings
- Evaluated using:
  - Confusion Matrix
  - Accuracy
  - Classification Report (Precision/Recall/F1)

## Files
- `Kindle Review Sentiment Analyis (1).ipynb` — main notebook (data → preprocessing → modeling → evaluation)


## Tech stack
- Python
- Pandas, NumPy
- NLTK (stopwords, lemmatizer)
- BeautifulSoup (HTML removal)
- scikit-learn (vectorizers, Naive Bayes, metrics)
- gensim (Word2Vec pretrained vectors)
- Jupyter Notebook

## How to run
```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate

pip install pandas numpy scikit-learn nltk beautifulsoup4 lxml gensim jupyter

jupyter notebook
