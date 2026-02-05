import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import joblib
from src.preprocess_text import clean_text
from src.sentiment import rating_to_sentiment
from src.recommender import RecommenderSystem


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/reviews_raw.csv")

df_small = df[['brand','categories','name','reviews.rating','reviews.text']].copy()
df_small['clean_text'] = df_small['reviews.text'].apply(clean_text)
df_small['sentiment'] = df_small['reviews.rating'].apply(rating_to_sentiment)

X = df_small['clean_text']
y = df_small['sentiment']

tfidf = TfidfVectorizer(max_features=20000, stop_words='english')
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": LinearSVC(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=150)
}

accuracies = {}
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    trained_models[name] = model
    print(f"{name} Accuracy: {acc}")

best_model_name = max(accuracies, key=accuracies.get)
best_model = trained_models[best_model_name]

joblib.dump(best_model, "models/sentiment_model.pkl")
joblib.dump(tfidf, "models/vectorizer_ml.pkl")

print(f"\nSaved Best Model: {best_model_name} — Accuracy: {accuracies[best_model_name]}")
