from fastapi import FastAPI
import joblib
import pandas as pd
import zipfile
import os
from src.preprocess_text import clean_text

app = FastAPI()

# --------------------------------------------------------
# EXTRACT CSV IF ZIP EXISTS
# --------------------------------------------------------
csv_path = "data/processed.csv"
zip_path = "data/processed.csv.zip"

if not os.path.exists(csv_path):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith(".csv"):
                    zip_ref.extract(file, "data/")
                    extracted_file = os.path.join("data", file)
                    if extracted_file != csv_path:
                        os.rename(extracted_file, csv_path)
    else:
        raise Exception("processed.csv.zip not found in /data folder!")


# --------------------------------------------------------
# LOAD DATA AND MODELS
# --------------------------------------------------------
df = pd.read_csv(csv_path)

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer_ml.pkl")

knn = joblib.load("models/knn.pkl")
X = joblib.load("models/tfidf_matrix.pkl")


# --------------------------------------------------------
# SENTIMENT PREDICTION ENDPOINT
# --------------------------------------------------------
@app.get("/sentiment")
def sentiment_api(text: str):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return {"sentiment": pred}


# --------------------------------------------------------
# PRODUCT RECOMMENDER ENDPOINT
# --------------------------------------------------------
@app.get("/recommend")
def recommend_api(index: int):
    if index < 0 or index >= len(df):
        return {"error": "Index out of range!"}

    distances, indices = knn.kneighbors(X[index], n_neighbors=6)
    products = df.iloc[indices[0][1:]][['name','brand','categories']]
    return products.to_dict(orient="records")
