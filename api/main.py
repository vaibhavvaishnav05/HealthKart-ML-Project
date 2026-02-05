from fastapi import FastAPI
import joblib
import pandas as pd
from src.preprocess_text import clean_text

app = FastAPI()

df = pd.read_csv("data/processed.csv")

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer_ml.pkl")

knn = joblib.load("models/knn.pkl")
X = joblib.load("models/tfidf_matrix.pkl")


@app.get("/sentiment")
def sentiment_api(text: str):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return {"sentiment_predicted": pred}


@app.get("/recommend")
def recommend_api(index: int):
    distances, indices = knn.kneighbors(X[index], n_neighbors=6)
    products = df.iloc[indices[0][1:]][['name','brand','categories']]
    return products.to_dict(orient="records")
