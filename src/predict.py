import joblib
from src.preprocess_text import clean_text

vectorizer = joblib.load("models/vectorizer.pkl")

def predict_sentiment(text):
    text = clean_text(text)
    x = vectorizer.transform([text])
    return x

example = "This product is amazing!"
print("Vectorized:", predict_sentiment(example))
