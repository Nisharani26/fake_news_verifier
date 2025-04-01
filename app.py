import os
import pickle
import json
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Define paths for models and vectorizers
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# Load models
models = {
    "english": pickle.load(open(os.path.join(MODEL_DIR, "english_news_model.pkl"), "rb")),
    "hindi": pickle.load(open(os.path.join(MODEL_DIR, "hindi_news_model.pkl"), "rb")),
    "tamil": pickle.load(open(os.path.join(MODEL_DIR, "tamil_news_model.pkl"), "rb"))
}

# Load TF-IDF vectorizers
vectorizers = {
    "english": pickle.load(open(os.path.join(MODEL_DIR, "tfidf_vectorizer_english.pkl"), "rb")),
    "hindi": pickle.load(open(os.path.join(MODEL_DIR, "tfidf_vectorizer_hindi.pkl"), "rb")),
    "tamil": pickle.load(open(os.path.join(MODEL_DIR, "tfidf_vectorizer_tamil.pkl"), "rb"))
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()
        text = data.get("text")
        language = data.get("language")  # "english", "hindi", or "tamil"

        if not text or not language:
            return jsonify({"error": "Invalid input, please provide 'text' and 'language'"}), 400

        if language not in models:
            return jsonify({"error": f"Language '{language}' not supported"}), 400

        # Preprocess text (use the trained vectorizer)
        vectorizer = vectorizers[language]
        text_vectorized = vectorizer.transform([text])

        # Predict using the selected model
        model = models[language]
        prediction = model.predict(text_vectorized)[0]

        # Return prediction
        return jsonify({"prediction": "Fake" if prediction == 1 else "Real"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
