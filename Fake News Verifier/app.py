import os
import pickle
from preprocessing.preprocess_tamil import preprocess_user_input as preprocess_tamil
from preprocessing.preprocess_hindi import preprocess_user_input as preprocess_hindi
from preprocessing.preprocess_english import preprocess_user_input as preprocess_english
import string
# Define project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_dir, "models")  # Adjusted path to point to the correct models folder

# Load models and vectorizers
try:
    models = {
        "english": pickle.load(open(os.path.join(model_dir, "english_news_model.pkl"), "rb")),
        "hindi": pickle.load(open(os.path.join(model_dir, "hindi_news_model.pkl"), "rb")),
        "tamil": pickle.load(open(os.path.join(model_dir, "tamil_news_model.pkl"), "rb"))
    }
    
    vectorizers = {
        "english": pickle.load(open(os.path.join(model_dir, "tfidf_vectorizer_english.pkl"), "rb")),
        "hindi": pickle.load(open(os.path.join(model_dir, "tfidf_vectorizer_hindi.pkl"), "rb")),
        "tamil": pickle.load(open(os.path.join(model_dir, "tfidf_vectorizer_tamil.pkl"), "rb"))
    }
    print("Models and vectorizers loaded successfully!")

except FileNotFoundError as e:
    print(f"Error loading model/vectorizer: {e}")
    exit(1)

# Preprocessing functions mapping
preprocess_funcs = {
    "english": preprocess_english,
    "hindi": preprocess_hindi,
    "tamil": preprocess_tamil
}

# Function to verify input text and predict if it's real or fake
def verify_news(input_text, language):
    # Check if the language is valid
    if language not in models or language not in vectorizers:
        return "Error: Language model or vectorizer not found!"

    # Step 1: Preprocess the user input in the selected language
    processed_input = preprocess_funcs[language](input_text)
    print(f"Processed user input: {processed_input}")

    # Step 2: Vectorize the processed input text
    vectorized_input = vectorizers[language].transform([processed_input])

    # Step 3: Make a prediction using the corresponding model
    prediction = models[language].predict(vectorized_input)
    probability = models[language].predict_proba(vectorized_input)[0][1]  # Get probability for 'Fake' class

    # Step 4: Return prediction result (Assuming 1 = Fake, 0 = Real) and the confidence score
    result = "Fake News" if prediction[0] == 1 else "Real News"
    return f"Prediction: {result}, Confidence (Probability): {probability * 100:.2f}%"

# Example user input from frontend
user_input = "This is an example of fake news about global warming!"
language = "english"  # This would be dynamically selected based on user input (e.g., from the frontend)

# Call the verification function with the user input and selected language
result = verify_news(user_input, language)

# Output the result
print(f"Original user input: {user_input}")
print(f"Prediction result: {result}")
