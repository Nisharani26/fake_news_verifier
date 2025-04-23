# import os
# import pandas as pd
# import re

# # Define file paths
# project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Construct paths to the dataset and output file
# data_path = os.path.join(project_dir, "dataset", "english", "Raw", "structured_data_english.csv")
# output_path = os.path.join(project_dir, "dataset", "english", "Processed", "processed_file_english.csv")

# # Define correct column names
# expected_columns = ["title", "text", "label"]

# try:
#     df = pd.read_csv(data_path, skiprows=1, names=expected_columns, on_bad_lines="skip")
#     print("✅ Dataset loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading dataset: {e}")
#     exit()

# # Handle missing values
# df.dropna(inplace=True)
# print("✅ Missing values removed!")

# # Remove duplicates
# df.drop_duplicates(subset=["title"], keep="first", inplace=True)
# print("✅ Duplicates removed!")

# # Clean unwanted characters
# df["title"] = df["title"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", str(x)).strip())
# df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())
# print("✅ Text cleaned!")

# # Remove short/irrelevant data
# df = df[df["title"].str.len() > 10]
# df = df[df["text"].str.len() > 20]
# print("✅ Short/irrelevant data removed!")

# # Save cleaned dataset
# df.to_csv(output_path, index=False)
# print(f"✅ English data preprocessing complete! ✅ Processed file saved at: {output_path}")

# import string  # Add this import at the top of your file

# def preprocess_user_input(user_input):
#     # Example preprocessing steps for English text
#     user_input = user_input.lower()  # Convert text to lowercase
#     user_input = user_input.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
#     user_input = ' '.join(user_input.split())  # Remove extra spaces
#     return user_input




#changes
import os
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Text Preprocessing 
def preprocess_user_input(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

# Load & Preprocess Dataset 
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "dataset", "english", "Raw", "structured_data_english.csv")
output_path = os.path.join(project_dir, "dataset", "english", "Processed", "processed_file_english.csv")

expected_columns = ["title", "text", "label"]

try:
    df = pd.read_csv(data_path, skiprows=1, names=expected_columns, on_bad_lines="skip")
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

df.dropna(inplace=True)
df.drop_duplicates(subset=["title"], keep="first", inplace=True)
df["title"] = df["title"].apply(preprocess_user_input)
df["text"] = df["text"].apply(preprocess_user_input)
df = df[df["title"].str.len() > 10]
df = df[df["text"].str.len() > 20]

df.to_csv(output_path, index=False)
print(f"✅ Processed file saved at: {output_path}")

# ========== 3. Class Distribution ==========
print("Label Distribution BEFORE balancing:")
print(df['label'].value_counts())

# ========== 4. Balance Dataset ==========
real_df = df[df['label'] == 1]  # Real
fake_df = df[df['label'] == 0]  # Fake

if len(fake_df) == 0:
    print("❌ No 'Fake' samples found. Cannot proceed.")
    exit()

# Upsample the minority class
fake_upsampled = fake_df.sample(n=len(real_df), replace=True, random_state=42)
df_balanced = pd.concat([real_df, fake_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Label Distribution AFTER balancing:")
print(df_balanced['label'].value_counts())

# Feature Extraction 
X = df_balanced["title"] + " " + df_balanced["text"]
y = df_balanced["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

#Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, stratify=y, random_state=42)

# Train Model 
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

#Evaluation
y_pred = model.predict(X_test)

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

# print("✅ Confusion Matrix:")
# cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
# disp.plot(cmap="Blues")
# plt.title("Confusion Matrix")
# plt.show()


#changes
