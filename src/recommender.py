import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class RecommenderSystem:
    def __init__(self):
        self.vectorizer = None
        self.knn = None

    def fit(self, texts):
        self.vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
        X = self.vectorizer.fit_transform(texts)

        self.knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn.fit(X)

        joblib.dump(self.vectorizer, "models/vectorizer.pkl")
        joblib.dump(self.knn, "models/knn.pkl")

        return X

    def recommend(self, X, index, df, n=5):
        distances, indices = self.knn.kneighbors(X[index], n_neighbors=n+1)
        return df.iloc[indices[0][1:]]
