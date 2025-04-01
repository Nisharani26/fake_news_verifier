import os
import pandas as pd
import re

# Define file paths
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct paths to the dataset and output file
data_path = os.path.join(project_dir, "dataset", "english", "Raw", "structured_data_english.csv")
output_path = os.path.join(project_dir, "dataset", "english", "Processed", "processed_file_english.csv")

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
print(f"✅ English data preprocessing complete! ✅ Processed file saved at: {output_path}")


def preprocess_user_input(user_input):
    """Preprocess English news text"""
    user_input = user_input.lower()
    user_input = re.sub(r'\d+', '', user_input)  # Remove numbers
    user_input = user_input.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    user_input = re.sub(r'\s+', ' ', user_input).strip()  # Remove extra spaces
    return user_input

