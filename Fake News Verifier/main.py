import pandas as pd
import re
import nltk
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Define file paths
data_file = r"C:\Users\vigne\Documents\fake_news_verifier\Fake News Verifier\English_kaggle\English_News.csv"
cleaned_file_path = r"C:\Users\vigne\Documents\fake_news_verifier\Fake News Verifier\English_kaggle\Cleaned_News_dataset.csv"

# Define correct column names
expected_columns = ["title", "text", "label"]

# Load dataset
try:
    df = pd.read_csv(data_file, skiprows=1, names=expected_columns, on_bad_lines="skip")
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Remove missing values & duplicates
df.dropna(inplace=True)
df.drop_duplicates(subset=["title"], keep="first", inplace=True)

# Clean text (remove special characters)
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces
    return text

df["title"] = df["title"].apply(clean_text)
df["text"] = df["text"].apply(clean_text)

# Remove short data
df = df[df["title"].str.len() > 10]
df = df[df["text"].str.len() > 20]

# Save cleaned dataset
df.to_csv(cleaned_file_path, index=False)
print(f"✅ Cleaned dataset saved at: {cleaned_file_path}")

# Text Preprocessing - Tokenization, Stopwords Removal, Lemmatization
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase & tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalnum()]
    return " ".join(tokens)

df["text"] = df["text"].apply(preprocess_text)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Convert Text to Vectors using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate Model
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save Model and Vectorizer for Deployment
with open("fake_news_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model and vectorizer saved for deployment!")
