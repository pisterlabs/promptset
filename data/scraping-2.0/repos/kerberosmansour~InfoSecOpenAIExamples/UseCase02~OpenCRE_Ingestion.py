import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from tqdm import tqdm, tqdm_pandas
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to check if a URL is valid
def is_valid_url(url):
    return url.startswith('http://') or url.startswith('https://')

# Connect to the SQLite database
conn = sqlite3.connect('db.sqlite')

# Load the unique, non-null, non-empty URLs with name and section columns from the 'node' table
records_df = pd.read_sql_query("SELECT DISTINCT link, name, section FROM node WHERE link IS NOT NULL AND link != '' AND link != 'n/a'", conn)

# Filter records with valid URLs
records_df['is_valid'] = records_df['link'].apply(lambda url: is_valid_url(url))
valid_records_df = records_df[records_df['is_valid']].drop(columns=['is_valid'])

# Function to get text content from a URL
def get_text_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return ' '.join(soup.stripped_strings)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content for URL: {url} - {str(e)}")
        return ''

# Fetch content for each URL and add it to a new 'Content' column
valid_records_df['Content'] = [get_text_content(url) for url in tqdm(valid_records_df['link'], desc="Fetching content")]

# Remove HTML or MD formatting and white spaces in the content
valid_records_df['Content'] = valid_records_df['Content'].str.strip().replace('\s+', ' ', regex=True)

# Function to generate embeddings using OpenAI
def generate_embeddings(text, model="text-embedding-ada-002", max_tokens=8191):
    truncated_text = text[:max_tokens]
    return openai.Embedding.create(input=[truncated_text], model=model)['data'][0]['embedding']

# Generate embeddings for the content and add it to a new 'ada_v2_embedding' column
tqdm.pandas(desc="Generating embeddings")
valid_records_df['ada_v2_embedding'] = valid_records_df['Content'].progress_apply(lambda x: generate_embeddings(x, model='text-embedding-ada-002'))

# Save the DataFrame as a Parquet file
valid_records_df.to_parquet('url_contents.openai.parquet')

# Close the database connection
conn.close()
