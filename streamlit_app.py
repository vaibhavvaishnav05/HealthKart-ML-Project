import streamlit as st
import pandas as pd
import joblib
from src.preprocess_text import clean_text
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os

# --------------------------------------------------------
# EXTRACT CSV IF MISSING (USING processed.zip)
# --------------------------------------------------------
csv_path = "data/processed.csv"
zip_path = "data/processed.zip"     # IMPORTANT: using THIS FILE

if not os.path.exists(csv_path):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith(".csv"):
                    zip_ref.extract(file, "data/")
                    extracted = os.path.join("data", file)
                    if extracted != csv_path:
                        os.rename(extracted, csv_path)
    else:
        st.error("❌ processed.zip not found in /data folder!")


# --------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------
df = pd.read_csv(csv_path)


# --------------------------------------------------------
# LOAD MODELS
# --------------------------------------------------------
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer_ml.pkl")
knn = joblib.load("models/knn.pkl")
X = joblib.load("models/tfidf_matrix.pkl")


# --------------------------------------------------------
# CATEGORY RECOMMENDER
# --------------------------------------------------------
def recommend_by_category(category, top_n=5):
    subset = df[df['categories'] == category]
    if len(subset) == 0:
        return pd.DataFrame({"Error": ["No products found in this category"]})
    subset = subset.sort_values("reviews.rating", ascending=False)
    return subset[['name', 'brand', 'categories', 'reviews.rating']].head(top_n)


# --------------------------------------------------------
# STREAMLIT SETTINGS
# --------------------------------------------------------
st.set_page_config(page_title="HealthKart Sentiment & Recommendation", layout="wide")
st.title("🛒 HealthKart – Sentiment Analysis & Product Recommendation System")

menu = st.sidebar.radio("Navigation", ["Sentiment Analysis", "Recommendations", "Insights Dashboard"])


# --------------------------------------------------------
# 1️⃣ SENTIMENT ANALYSIS
# --------------------------------------------------------
if menu == "Sentiment Analysis":
    st.header("🔍 Sentiment Analysis from Review Text")

    user_text = st.text_area("Enter your product review:", height=150)

    if st.button("Predict Sentiment"):
        if user_text.strip() == "":
            st.error("Please enter some text")
        else:
            cleaned = clean_text(user_text)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            st.success(f"Predicted Sentiment: **{pred}**")


# --------------------------------------------------------
# 2️⃣ CATEGORY BASED RECOMMENDATIONS
# --------------------------------------------------------
if menu == "Recommendations":
    st.header("✨ Category-Based Product Recommendations")

    category_list = sorted(df['categories'].dropna().unique())
    selected_category = st.selectbox("Choose a Category:", category_list)

    if st.button("Get Recommendations"):
        results = recommend_by_category(selected_category)
        st.write("### Recommended Products:")
        st.table(results)


# --------------------------------------------------------
# 3️⃣ INSIGHTS DASHBOARD
# --------------------------------------------------------
if menu == "Insights Dashboard":

    st.header("📊 Insights & Visualizations")

    st.subheader("Sentiment Distribution")
    fig1 = plt.figure(figsize=(6, 3))
    sns.countplot(x=df['sentiment'], palette="viridis")
    st.pyplot(fig1)

    st.subheader("Rating Distribution")
    fig2 = plt.figure(figsize=(6, 3))
    sns.countplot(x=df['reviews.rating'], palette="coolwarm")
    st.pyplot(fig2)

    st.subheader("WordCloud of Reviews")
    df['clean_text'] = df['clean_text'].fillna("").astype(str)
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df['clean_text']))
    fig3 = plt.figure(figsize=(10, 4))
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(fig3)

    st.subheader("Top Brands by Positive Sentiment")
    brand_pos = df[df['sentiment'] == "Positive"]['brand'].value_counts().head(10)
    fig4 = plt.figure(figsize=(10, 3))
    brand_pos.plot(kind='bar', color='green')
    plt.ylabel("Positive Reviews Count")
    st.pyplot(fig4)
