import os
import pandas as pd
from indicnlp.tokenize import indic_tokenize
import string

# Determine the absolute path of the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct paths to the dataset, stopwords file, and output file
data_path = os.path.join(project_dir, "dataset", "tamil", "Raw", "structured_data_tamil.csv")
stopwords_path = os.path.join(project_dir, "dataset", "tamil", "Raw", "tamil_stopwords_(final).csv")
output_path = os.path.join(project_dir, "dataset", "tamil", "Processed", "processed_file_tamil.csv")

# Verify that the dataset file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at: {data_path}")

# Verify that the stopwords file exists
if not os.path.exists(stopwords_path):
    raise FileNotFoundError(f"Stopwords file not found at: {stopwords_path}")

# Load the dataset
df = pd.read_csv(data_path)

# Tokenize the 'News' column
df["tokenized_text"] = df["News"].apply(lambda text: indic_tokenize.trivial_tokenize(text, lang='ta'))

# Load Tamil stopwords
with open(stopwords_path, "r", encoding="utf-8") as f:
    tamil_stopwords = set(f.read().splitlines())

# Function to remove stopwords from tokenized text
def remove_stopwords(tokenized_list):
    return [word for word in tokenized_list if word not in tamil_stopwords]

# Apply stopwords removal
df["stopwords_removed"] = df["tokenized_text"].apply(remove_stopwords)

# Function to remove punctuation from text
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Convert list of words back to string and remove punctuation
df["standardized_text"] = df["stopwords_removed"].apply(lambda x: remove_punctuation(' '.join(x)))

# Save the processed DataFrame to a CSV file
df.to_csv(output_path, index=False, encoding="UTF-8")

print(f"Tamil data preprocessing complete! ✅ Processed file saved at: {output_path}")
