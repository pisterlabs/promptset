import openai
import pandas as pd
import numpy as np
import configparser
from getpass import getpass
from openai.embeddings_utils import get_embedding

# Read config
config = configparser.ConfigParser()
config.read('config.ini')

# Write KEY for openAI API
openai.api_key = config['openai']['api_key']

# Read the data from the csv file
df = pd.read_csv('.\Data\word_embeddings.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

# Input a search term
search_term = input('Enter a search term: ')
search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")

# Calculate the cosine similarity between the search term and each tag
df['similarity'] = df['embedding'].apply(lambda x: np.dot(x, search_term_vector) / (np.linalg.norm(x) * np.linalg.norm(search_term_vector)))

# Sort the tags by similarity
df = df.sort_values(by='similarity', ascending=False)

# Print the top 10 tags
print(df.head(10))

# Save the results to a CSV file
df.to_csv('.\Data\search_results.csv')