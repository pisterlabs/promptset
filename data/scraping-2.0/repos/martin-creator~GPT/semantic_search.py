import os
import openai
from openai.embeddings_utils import get_embedding,cosine_similarity
import click
import pandas as pd
import numpy as np
from numpy.linalg import norm

# Simple semantic search
# Click documentation: https://click.palletsprojects.com/en/8.1.x/


def init_api():
    ''' Load API key from .env file'''
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ["API_KEY"]
    openai.organization = os.environ["ORG_ID"]


init_api()


df = pd.read_csv('words.csv') # This line creates a pandas dataframe from the csv file

# print(df.tail(5)) # This line prints the last 5 rows of the dataframe

# print(get_embedding("Hello", engine="text-embedding-ada-002"))

# get embedding for each word in the dataframe
df['embedding'] = df['Text'].apply(lambda x: get_embedding(x, engine="text-embedding-ada-002"))

df.to_csv('embeddings.csv')

# Convert the values in 'embedding' column to string representations
df['embedding'] = df['embedding'].astype(str)

# convert last column to numpy array
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

# get the search term from the user
user_search = input("Enter a search term: ")

# get the embedding for the search term
search_term_embedding = get_embedding(user_search, engine="text-embedding-ada-002")

#print(search_term_embedding)

# calculate the similarity between the search term and each word in the dataframe
df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, search_term_embedding))

# sort the dataframe by the similarity score
df = df.sort_values(by="similarity", ascending=False)

# print the top 5 results
#print(df.head(5))

print(df.head(5))
print(df.tail(5))

#print(df)

# Detailed description of cosine similarity:
# https://en.wikipedia.org/wiki/Cosine_similarity


# Cosine similarity is a way of measuring how similar two vectors are. It looks at the angle between
# two vectors (lines) and compares them. Cosine similarity is the cosine of the angle between the
# vector. A result is a number between -1 and 1. If the vectors are the same, the result is 1. If the
# vectors are completely different, the result is -1. If the vectors are at a 90-degree angle, the result is
# 0. In mathematical terms, this is the equation:
# $$
# Similarity = (A.B) / (||A||.||B||)
# $$
# • A and B are vectors
# • A.B is a way of multiplying two sets of numbers together. It is done by taking each number
# in one set and multiplying it with the same number in the other set, then adding all of those
# products together.
# • ||A|| is the length of the vector A. It is calculated by taking the square root of the sum of the
# squares of each element of the vector A.
# Let’s consider vector A = [2,3,5,2,6,7,9,2,3,4] and vector B = [3,6,3,1,0,9,2,3,4,5].
# This is how we can get the cosine similarity between them using Python:


# define two vectors
A = np.array([2,3,5,2,6,7,9,2,3,4])
B = np.array([3,6,3,1,0,9,2,3,4,5])

# calculate the dot product
dot_product = np.dot(A,B)

# calculate the length of each vector
A_length = norm(A)
B_length = norm(B)

# calculate the cosine similarity
cosine_similarity = dot_product / (A_length * B_length)

print( f"cosine_similarity: { cosine_similarity}")





