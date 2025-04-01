import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define file paths
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_data_path = os.path.join(project_dir, "dataset", "tamil", "Processed", "processed_file_tamil.csv")
model_dir = os.path.join(project_dir, "models")
os.makedirs(model_dir, exist_ok=True)  # Ensure model directory exists

vectorizer_save_path = os.path.join(model_dir, "tfidf_vectorizer_tamil.pkl")
model_save_path = os.path.join(model_dir, "tamil_news_model.pkl")

# Load preprocessed dataset
df = pd.read_csv(processed_data_path)

# Extract text and labels
tamil_texts = df["standardized_text"].astype(str)
y = df["Label"]

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(tamil_texts)

# Save the TF-IDF vectorizer
with open(vectorizer_save_path, "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize models
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
svm_model = SVC(random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

# Train models
log_reg_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Evaluate models
models = {
    "Logistic Regression": log_reg_model,
    "SVM": svm_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

# Ensemble Model
ensemble_model = VotingClassifier(
    estimators=[
        ('logistic_regression', log_reg_model),
        ('svm', svm_model),
        ('random_forest', rf_model),
        ('gradient_boosting', gb_model)
    ],
    voting='hard'
)

ensemble_model.fit(X_train, y_train)

# Evaluate Ensemble Model
y_pred_ensemble = ensemble_model.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Model Accuracy: {accuracy_ensemble * 100:.2f}%")
print("Ensemble Model Classification Report:")
print(classification_report(y_test, y_pred_ensemble))

# Save the trained model
with open(model_save_path, "wb") as model_file:
    pickle.dump(ensemble_model, model_file)

print(f"\n✅ Model saved at: {model_save_path}")
print(f"✅ TF-IDF Vectorizer saved at: {vectorizer_save_path}")
