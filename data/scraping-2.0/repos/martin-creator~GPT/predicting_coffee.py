# When a user enters the name of a coffee, we will use the OpenAI Embeddings API to get the
# embedding for the review text of that coffee. Then, we will calculate the cosine similarity between
# the input coffee review and all other reviews in the dataset. The reviews with the highest cosine
# similarity scores will be the most similar to the input coffee’s review. We will then print the names
# of the most similar coffees to the user.

from lib2to3.pgen2 import token
import os
import openai
import pandas as pd
import numpy as np
import nltk
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

# develop a command-line tool that can assist us with Linux commands through conversation.
# Click documentation: https://click.palletsprojects.com/en/8.1.x/


def init_api():
    ''' Load API key from .env file'''
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ["API_KEY"]
    openai.organization = os.environ["ORG_ID"]



# download the dataset

def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/stopwords")
    except LookupError:
        nltk.download("stopwords")


def preprocess_review(review):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    stopwords = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(review.lower())
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)

# stemming is the  process of reducing inflected (or sometimes derived) words to their word stem, base or root form

init_api()

# download_nltk_data()
download_nltk_data()

# Read user input
input_coffee_name = input("Enter a coffee name: ")

# Load the CSV fiele into pandas dataframe (only the first 50 rows for now to speed up the demo and avoid paying for too many API calls)
df = pd.read_csv("simplified_coffee.csv", nrows=50)

# Preprocess the review text: lowercase, tokenise, remove stopwords, and stem
df["preprocessed_review"] = df["review"].apply(preprocess_review)

# Get the embeddings for each review
review_embeddings = []

for review in df["preprocessed_review"]:
    review_embeddings.append(get_embedding(review, engine="text-embedding-ada-002"))

# Get the embedding for the input coffee name
try:
    print("Getting embedding for input coffee name...")
    # input_coffee_index = df[df['name'] == input_coffee_name].index[0]
    input_coffee_index = df[df['name'].str.contains(input_coffee_name, case=False)].index[0]
    print(input_coffee_index)
except:
    print("Sorry, we dont have that coffee in our database. Please try again.")
    exit()

# 1. df['name'] == input_coffee_name creates a boolean mask that is True for rows where the
# “name” column is equal to input_coffee_name and False for all other rows.
# 2. df[df['name'] == input_coffee_name] uses this boolean mask to filter the DataFrame and
# returns a new DataFrame that contains only the rows where the “name” column is equal to
# input_coffee_name.
# 3. df[df['name'] == input_coffee_name].index returns the index labels of the resulting filtered
# DataFrame.
# 4. index[0] retrieves the first index label from the resulting index labels. Since the filtered
# DataFrame only contains one row, this is the index label for that row.


# Calculate the cosine similarity between the input coffee review and all other reviews

similarities = []
input_review_embedding = review_embeddings[input_coffee_index]
print(input_review_embedding)

for review_embedding in review_embeddings:
    similarity = cosine_similarity(input_review_embedding, review_embedding)
    similarities.append(similarity)

print("The similarity scores are: ")
print(similarities)


# Get the indices of the most similar reviews (excluding the input review itself)
most_similar_indices = np.argsort(similarities)[::-1][1:6]
print("The most similar reviews are: ")
print(most_similar_indices)


# Get the names of the most similar coffees
similar_coffee_names = df.iloc[most_similar_indices]["name"].tolist()

# Print the names of the most similar coffees
for coffee_name in similar_coffee_names:
    print(coffee_name)