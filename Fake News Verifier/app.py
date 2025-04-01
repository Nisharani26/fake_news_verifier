import os
import pickle
from preprocessing.preprocess_tamil import preprocess_user_input as preprocess_tamil
from preprocessing.preprocess_hindi import preprocess_user_input as preprocess_hindi


# Define project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_dir,"models")  # Adjusted path to point to the correct models folder

# Load models and vectorizers
models = {
    "english": pickle.load(open(os.path.join(model_dir, "english_news_model.pkl"), "rb")),  # Corrected file path
    "hindi": pickle.load(open(os.path.join(model_dir, "hindi_news_model.pkl"), "rb")),
    "tamil": pickle.load(open(os.path.join(model_dir, "tamil_news_model.pkl"), "rb"))
}

vectorizers = {
    "english": pickle.load(open(os.path.join(model_dir, "tfidf_vectorizer_english.pkl"), "rb")),
    "hindi": pickle.load(open(os.path.join(model_dir, "tfidf_vectorizer_hindi.pkl"), "rb")),
    "tamil": pickle.load(open(os.path.join(model_dir, "tfidf_vectorizer_tamil.pkl"), "rb"))
}

# Check if models and vectorizers are loaded successfully
print("Loaded Models:", models.keys())
print("Loaded Vectorizers:", vectorizers.keys())

# Example user input from frontend
user_input = "உலகின் மிகப்பெரிய எண்ணெய் கசிவானது!!!"

# Preprocess the user input
processed_user_input = preprocess_tamil(user_input)

# Print both original and preprocessed text
print(f"Original user input: {user_input}")
print(f"Processed user input: {processed_user_input}")