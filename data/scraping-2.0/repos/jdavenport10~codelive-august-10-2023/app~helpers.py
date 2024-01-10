import openai
import numpy as np
import json
from .prompt import *

openai.api_key = json.load(open("openai.json"))["key"]

embeddings_file = open("embeddings-backup.json", "r")
database = json.loads(embeddings_file.read())
vector_matrix = np.array([v["vector"] for _, v in database.items()])

def embed_query(query_string):
    print(f"EMBEDDING QUERY STRING: {query_string}")
    query_embedding = openai.Embedding.create(
        input=query_string, 
        model="text-embedding-ada-002"
    )['data'][0]['embedding']

    return query_embedding

def search(query_string):
    query_embed = embed_query(query_string)
    print("DOING MATH")
    # IMPORTANT: We can use Dot Product here because the vectors are normalized
    # IMPORTANT: For unnormalized vectors, you must use a different metric (cos_similarity)
    # Calculate distances for all vectors in our database

    similarities = np.dot(query_embed, vector_matrix.T)

    # Retrieve the top 10 "most similar"
    top_10 = (np.argpartition(similarities, -10)[-10:])

    # Get the original text from our database for each vector
    context = [database[str(i+1)]["text"] for i in top_10]

    # return a single string as the context for our query
    print(context)
    return " | ".join(context)

def chat_gpt_query(query_string, existing_context=[]):
    print("GETTING CONTEXT")
    # Find our context to send to ChatGPT
    context = search(query_string)

    # Assemble our Prompt
    query_prompt = prompt.format(
        context=context,
        user_query=query_string
    )
    chatdata = existing_context + [
        {"role": "user", "content": query_prompt}
    ] 
    print("ASKING CHATGPT")

    # Query ChatGPT for our response
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=chatdata,
        temperature=0.0
    )["choices"][0]["message"]["content"]
    print(resp)
    return resp