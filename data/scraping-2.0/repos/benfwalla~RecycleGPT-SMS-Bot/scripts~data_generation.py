import openai
import os
import pandas as pd
import re
import requests
import tiktoken
from bs4 import BeautifulSoup


EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-4"
ENC = tiktoken.encoding_for_model(COMPLETIONS_MODEL)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 100)


def create_embedding(text, openai_api_key):
  openai.api_key = openai_api_key
  response = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
  embedding = response.data[0].embedding
  return embedding


def clean_text(text):
  # Convert text to lowercase
  text = text.lower()

  # Remove newline characters
  text = text.replace('\n', ' ').replace('\r', ' ')

  # Remove URLs
  text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

  # Remove extra spaces
  text = re.sub(r'\s+', ' ', text).strip()

  return text


def count_tokens(text):
  tokens = ENC.encode(text)
  tokens_count = len(tokens)
  return tokens_count


def extract_recycling_info(url):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  sections = soup.find_all('div', class_="page-section-container")

  # Initialize the DataFrame
  data = []

  for section in sections:
    # Find all the <p>, <ul>, and <ol> tags
    tags = section.find_all(['p', 'ul', 'ol'])

    # Iterate over the tags and extract the text and the last <h1> to <h5> tag that preceded it
    last_heading = None
    for tag in tags:
      if tag.find_previous(['h1', 'h2', 'h3']):
        last_heading = tag.find_previous(['h1', 'h2', 'h3']).text.strip()
      text = tag.text.strip()
      if text:
        data.append([last_heading, text])

  # Create the DataFrame
  df = pd.DataFrame(data, columns=['Heading', 'Text'])

  # Combine the Heading and Text
  df['Heading and Text'] = df['Heading'] + ' - ' + df['Text']

  # Clean the Heading and Text of unwanted characters
  df['Heading and Text'] = df['Heading and Text'].apply(clean_text)

  # Get the token count for each row
  df['Number of Tokens'] = df['Heading and Text'].apply(count_tokens)
  df = df[df['Number of Tokens'] >= 15]

  # Create the embedding for each row
  df['Embedding'] = df['Heading and Text'].apply(create_embedding)

  return df


# url = 'https://www.denvergov.org/Government/Agencies-Departments-Offices/Agencies-Departments-Offices-Directory/Recycle-Compost-Trash/Recycle'
# recycling_df = extract_recycling_info(url)
# recycling_df.to_csv('co_recycling_data_with_embeddings.csv')
# print(recycling_df)
