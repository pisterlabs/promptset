import openai
import os
from dotenv import load_dotenv
import json
import numpy as np
from rich.prompt import Prompt
from annoy import AnnoyIndex

load_dotenv()
openai.api_key = os.environ.get('API_KEY')

# Load the embeddings from the file
with open("tmp/tweets_db.json", "r") as f:
    embeddings = json.load(f)

# Build the Annoy index
index = AnnoyIndex(len(embeddings[0]["embedding"]), metric="angular")
for i, item in enumerate(embeddings):
    index.add_item(i, item["embedding"])
index.build(50)  # 50 trees

while True:
    query = Prompt.ask("You")

    # Calculate the embedding for the query text
    response = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=query
    )
    query_embedding = response["data"][0]["embedding"]

    # Search for the most similar embeddings
    similarities = index.get_nns_by_vector(query_embedding, 10, include_distances=True)

    print("RESULTS:")
    # Print the top 3 most similar texts
    for i in range(10):
        # print(similarities[i]["text"], similarities[i]["similarity"])
        print(embeddings[similarities[0][i]]["text"], similarities[1][i])
