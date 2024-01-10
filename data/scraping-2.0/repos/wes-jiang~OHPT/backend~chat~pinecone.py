import featureform as ff
from featureform import local
import io
from io import StringIO
import pandas as pd
from sentence_transformers import SentenceTransformer
# import dotenv
import os
import openai
from cs161_scrape.py import *


pinecone = ff.register_pinecone(
    name="pinecone",
    project_id="59c003b",
    environment="northamerica-northeast1-gcp",
    api_key="ac43b8c2-f21e-4c29-bbb2-e8f5880175af",
)

@ff.entity
class Text_String:
    excerpt_embeddings = ff.Embedding(
        vectorize_excerpts[["PK", "Vector"]],
        dims=384,
        vector_db=pinecone,
        description="Embeddings from excerpts of chapters",
        variant="v1"
    )
    excerpts = ff.Feature(
        combine_dfs[["PK", "Text"]],
        type=ff.String,
        description="Excerpts' original text",
        variant="v1"
    )

@ff.ondemand_feature(variant="ohpt")
def relevant_excerpts(client, params, entity):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    search_vector = model.encode(params["query"])
    # print(search_vector)
    res = client.nearest("excerpt_embeddings", "v1", search_vector, k=5)
    return res

# creates the improved and contextualized prompt
@ff.ondemand_feature(variant="ohpt")
def contextualized_prompt(client, params, entity):
    pks = client.features([("relevant_excerpts", "ohpt")], {}, params=params)
    # print(pks)
    prompt = "Use the following pages from our textbook to answer the following question\n"
    for pk in pks[0]:
        prompt += "```"
        # print(client.features([("excerpts", "v1")], {"excerpt": pk}))
        prompt += client.features([("excerpts", "v1")], {"excerpt": pk})[0]
        prompt += "```\n"
    prompt += "Question: "
    prompt += params["query"]
    prompt += "?"
    return prompt

if __name__ == '__main__':
    className = '170'
    q = 'dynamic programming'
    main()