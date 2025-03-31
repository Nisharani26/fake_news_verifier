import pandas as pd
import re

# Define file paths
data_file = r"C:\Users\vigne\Documents\fake_news_verifier\Fake News Verifier\English_kaggle\English_News.csv"
cleaned_file_path = r"C:\Users\vigne\Documents\fake_news_verifier\Fake News Verifier\English_kaggle\Cleaned_News_dataset.csv"

# Define correct column names
expected_columns = ["title", "text", "label"]

try:
    # Load dataset and enforce column names
    df = pd.read_csv(data_file, skiprows=1, names=expected_columns, on_bad_lines="skip")
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Display first 5 rows
print(df.head())
print("Columns in dataset:", df.columns)

# Handle missing values
df.dropna(inplace=True)
print("✅ Missing values removed!")

# Remove duplicate news titles
df.drop_duplicates(subset=["title"], keep="first", inplace=True)
print("✅ Duplicates removed!")

# Clean unwanted characters
df["title"] = df["title"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", str(x)).strip())
df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())
print("✅ Text cleaned!")

# Remove short or irrelevant data
df = df[df["title"].str.len() > 10]
df = df[df["text"].str.len() > 20]
print("✅ Short/irrelevant data removed!")

# Save the cleaned dataset
df.to_csv(cleaned_file_path, index=False)
print(f"✅ Cleaned dataset saved at: {cleaned_file_path}")
