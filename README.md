# HealthKart – Sentiment + Brand Analysis + Recommendation System

This project analyzes customer reviews and builds:
✔ Sentiment analysis  
✔ Brand & category extraction  
✔ Memory-safe recommendation system (TF-IDF + KNN)  
✔ API endpoints for sentiment & recommendations  
✔ Dockerized deployment  

### Folder Structure
See complete project structure inside this repo.

---

## 🚀 1. Install Requirements
pip install -r requirements.txt

## 🚀 2. Run Training
python src/train.py

## 🚀 3. Run Predictions
python src/predict.py

## 🚀 4. Start API
uvicorn api.main:app --reload

## 🚀 5. Docker Build
docker build -t healthkart-app .

## 🚀 6. Docker Run
docker run -p 8000:8000 healthkart-app

---

## API Routes

### 1️⃣ Sentiment Prediction
POST /sentiment  
Body: {"text": "Product is very good!"}

### 2️⃣ Product Recommendation
POST /recommend  
Body: {"index": 10}

---

## Author
Vaibhav
