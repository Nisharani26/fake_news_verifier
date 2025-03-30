
# 1-- Load the dataset

import pandas as pd
import re

# Load the dataset
df = pd.read_csv(r"C:\Users\vigne\Documents\fake_news_verifier\Fake News Verifier\English\english_dataset.csv")


# Show first 5 rows
print(df.head())



# 2--  Check & Handle Missing Values

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

# Check again
print("After removing missing values:\n", df.isnull().sum())

# 3--  Remove Duplicates

# Remove duplicate news titles
df = df.drop_duplicates(subset=["title"], keep="first")

# Check if duplicates still exist
print("Duplicates left:", df.duplicated().sum())


#   4-- Clean Unwanted Characters


# Remove special characters and extra spaces
df["title"] = df["title"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", str(x)).strip())
df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())


#  5-- Remove Short or Irrelevant Data

# Remove very short titles and text
df = df[df["title"].str.len() > 10]
df = df[df["text"].str.len() > 20]


#  6-- Save the Cleaned Data

# Define the desired file path
csv_filename = r"C:\Users\vigne\Documents\fake_news_verifier\Fake News Verifier\English\Cleaned_news_dataset.csv"

# Save dataset to CSV
df.to_csv(csv_filename, index=False)

print("✅ Data cleaning done! Saved as 'Cleaned_News_Dataset.csv'")


# Convert labels: Real → 0, Fake → 1
df["label"] = df["label"].map({"Real": 0, "Fake": 1})

cleaned_Dataset_File_new = r"C:\Users\vigne\Documents\fake_news_verifier\Fake News Verifier\English\Cleaned_News_dataset_ready.csv"

# Save the cleaned dataset
df.to_csv(cleaned_Dataset_File_new, index=False)

print("✅ Labels updated! 'Real' → 0 and 'Fake' → 1")




