import os
import openai
import numpy as np
from dotenv import load_dotenv
#from datetime import datetime


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

#Index + 1 for tweets because SQL serial.


def compute_cosine_similarity(query_vector, vector_list):
    # Compute the dot product between the query vector and the vector list
    dot_product = np.dot(query_vector, vector_list.T)

    # Compute the cosine similarity between the query vector and the vector list
    similarity_scores = dot_product[0]

    # Get the indices that would sort the similarity scores in descending order
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Sort the similarity scores in descending order
    sorted_scores = similarity_scores[sorted_indices]

    # Sort the vector list by the similarity scores
    sorted_vectors = vector_list[sorted_indices]

    # Return the similarity scores, vector list, and the corresponding indices
    return sorted_scores, sorted_vectors, sorted_indices

def similarity_tweet(query_str, candidates_id=None, limit=20, vector_list=None):
    query_vector = np.array(openai.Embedding.create(input=query_str, model='text-embedding-ada-002')['data'][0]['embedding'])
    #now = datetime.now()
    query_vector = query_vector.reshape(1, -1)
    vector_list_2 = vector_list
    #print(vector_list)
    if candidates_id != None:
        vector_list_2  = vector_list[candidates_id]
    sorted_scores, _, sorted_indices = compute_cosine_similarity(query_vector, vector_list_2)
    #d = datetime.now() - now
    #print(d)
    if candidates_id != None:
        sorted_candidates_id = np.array(candidates_id)[sorted_indices]
        return list(sorted_candidates_id[:limit])
    return list(sorted_indices[:limit])



