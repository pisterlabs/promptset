# GPT3
import pandas as pd
import openai, numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt

# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def GPT3_embeddings(text, engine="text-similarity-davinci-001"):

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]


# embedding = get_embedding("Sample query text goes here", engine="text-search-ada-query-001")
# print(len(embedding))

# def getGPT3embeddings(judgement,timestamps_list):
#     #get embeddings of the judgement section
#     section_embedding = gpt3_embeddings(judgement,engine="text-search-ada-query-001" )
#     #get embeddings of the timestamps
#     timestamps_embeddings = [] 
#     for t in timestamps_list:
#         timestamps_embeddings.append(gpt3_embeddings(t,engine="text-search-ada-query-001"))
