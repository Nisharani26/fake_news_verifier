import os
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define file paths
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct paths to the dataset, stopwords file, and output file
data_path = os.path.join(project_dir, "dataset", "english", "Raw", "structured_data_english.csv")
output_path = os.path.join(project_dir, "dataset", "english", "Processed", "processed_file_english.csv")

# model_save_path = os.path.join(model_dir, "svm_fake_news_model.pkl")
# vectorizer_save_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

# Define correct column names
expected_columns = ["title", "text", "label"]

try:
    df = pd.read_csv(data_path, skiprows=1, names=expected_columns, on_bad_lines="skip")
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Handle missing values
df.dropna(inplace=True)
print("✅ Missing values removed!")

# Remove duplicates
df.drop_duplicates(subset=["title"], keep="first", inplace=True)
print("✅ Duplicates removed!")

# Clean unwanted characters
df["title"] = df["title"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", str(x)).strip())
df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())
print("✅ Text cleaned!")

# Remove short/irrelevant data
df = df[df["title"].str.len() > 10]
df = df[df["text"].str.len() > 20]
print("✅ Short/irrelevant data removed!")

# Save cleaned dataset
df.to_csv(output_path, index=False)
print(f"✅ Cleaned dataset saved at: {output_path}")

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
model = SVC(kernel="linear")
model.fit(X_train_tfidf, y_train)
print("✅ SVM model trained!")

# Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred) * 100  # Accuracy as percentage
print(f"\n✅ Model Accuracy: {accuracy:.2f}%")


# Save model and vectorizer
# with open(model_save_path, "wb") as model_file:
#     pickle.dump(model, model_file)
# with open(vectorizer_save_path, "wb") as vectorizer_file:
#     pickle.dump(vectorizer, vectorizer_file)


# print(f"\n✅ Model saved at: {model_save_path}")
# print(f"✅ TF-IDF Vectorizer saved at: {vectorizer_save_path}")
