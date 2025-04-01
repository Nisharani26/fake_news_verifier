import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define file paths
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_data_path = os.path.join(project_dir, "dataset", "english", "Processed", "processed_file_english.csv")
model_dir = os.path.join(project_dir, "models")
os.makedirs(model_dir, exist_ok=True)  # Ensure model directory exists

model_save_path = os.path.join(model_dir, "english_news_model.pkl")  # Fixed path
vectorizer_save_path = os.path.join(model_dir, "tfidf_vectorizer_english.pkl")  # Fixed path

# Load preprocessed dataset
try:
    df = pd.read_csv(processed_data_path)
    print("✅ Processed dataset loaded successfully!")
    
    # Validate required columns
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"❌ Missing columns in dataset! Required: {required_columns}, Found: {df.columns}")

except Exception as e:
    print(f"❌ Error loading processed dataset: {e}")
    exit()

# Split dataset into train and test sets
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split into train and test sets!")

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("✅ Text vectorized using TF-IDF!")

# Train SVM model
model = SVC(kernel="linear", random_state=42)  # Added random_state
model.fit(X_train_tfidf, y_train)
print("✅ SVM model trained!")

# Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred) * 100  # Accuracy as percentage
print(f"\n✅ Model Accuracy: {accuracy:.2f}%")

# Save model and vectorizer
with open(model_save_path, "wb") as model_file:
    pickle.dump(model, model_file)
with open(vectorizer_save_path, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print(f"\n✅ Model saved at: {model_save_path}")
print(f"✅ TF-IDF Vectorizer saved at: {vectorizer_save_path}")
