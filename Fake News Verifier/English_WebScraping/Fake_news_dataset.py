import requests
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate  # For table formatting
import re  # Import regular expressions for filtering

# Define a news website (Example: BBC News)
url = "https://www.snopes.com/category/politics/"

# Send a request to get the webpage content
response = requests.get(url)

# Check if the request was successful
if response.status_code != 200:
    print("Failed to fetch news articles. Exiting.")
    exit()

soup = BeautifulSoup(response.text, "html.parser")

# Find all possible news headlines from multiple tags
headline_tags = ["h1", "h2", "h3", "a", "span"]
articles = []

for tag in headline_tags:
    articles.extend(soup.find_all(tag))  # Collect all possible headlines

data = []

# Define regex pattern to filter unwanted titles (timestamps, numbers, etc.)
# invalid_title_pattern = r"^\d{1,2} mins ago$|^\d{1,2} hours ago$|^\d{4}$|^\d+\s"

for article in articles:
    title = article.text.strip()  # Extract text
    link_tag = article.find_parent("a")  # Find parent <a> tag for link

    # Extract link if available, otherwise mark as "No Link"
    link = link_tag["href"] if link_tag and "href" in link_tag.attrs else "No Link"

    # If relative link, add base URL
    full_link = f"https://www.snopes.com{link}" if link.startswith("/") else link

    # Try fetching full article text
    try:
        news_page = requests.get(full_link)
        news_soup = BeautifulSoup(news_page.text, "html.parser")
        text = news_soup.find("p").text if news_soup.find("p") else "No content"
    except:
        text = "Error fetching article"

    # Apply filtering conditions:
    if (
        title  # Title should not be empty
        and text not in ["No content", "Error fetching article"]  # Ignore missing content
        and link != "No Link"  # Ensure valid links
        and not re.match(invalid_title_pattern, title)  # Exclude unwanted titles
    ):
        data.append({"title": title, "text": text, "label": "Fake"})  # Modify label accordingly

# Convert to DataFrame
df = pd.DataFrame(data)

# Define the desired file path
csv_filename = r"C:\Users\vigne\Documents\fake_news_verifier\Fake News Verifier\English_WebScraping\Fake_filtered_news_dataset.csv"

# Save dataset to CSV
df.to_csv(csv_filename, index=False)



# Print dataset as a formatted table (only valid rows)
if not df.empty:
    print(tabulate(df, headers="keys", tablefmt="grid"))
    print(f"\n Dataset saved as '{csv_filename}' and created successfully!")
else:
    print("\n No valid news articles found!")
