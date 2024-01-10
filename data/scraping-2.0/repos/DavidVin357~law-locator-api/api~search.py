import pinecone
import openai

import os
from dotenv import load_dotenv

load_dotenv()

emb_model_name = os.getenv("EMBEDDING_MODEL_NAME")
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(
    api_key=os.getenv("PINECONE_KEY"),
    environment=os.getenv("PINECONE_ENV"),  # find next to API key in console
)

import json


def search(query):
    index = pinecone.Index("openai")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """You are given a query about some legal matter.
                You need to convert it into a query suitable for search with ada-002 embeddings model.
                Don't try to specify the legislature or any other additional attributes, just convert the bare question.
                You are recommended to augment initial query if it can help. Return the result.
                  """,
            },
            {"role": "user", "content": query},
        ],
    )

    emb_query = response["choices"][0]["message"]["content"]

    print("emb_query: ", emb_query)
    xq = openai.Embedding.create(input=emb_query, model=emb_model_name)["data"][0][
        "embedding"
    ]
    query_result = index.query([xq], top_k=5, include_metadata=True)

    matches = []

    for m in query_result["matches"]:
        article_id = m["id"].split("|")[0]
        paragraph_id = m["id"].split("|")[1]
        paragraph = m["metadata"]["text"]
        paragraph_title = m["metadata"]["title"]

        matches.append(
            {
                "article_id": article_id,
                "paragraph_id": paragraph_id,
                "paragraph_title": paragraph_title,
                "paragraph": paragraph,
            }
        )
    return matches


def get_answer(query: str, paragraphs: list):
    paragraphs_content = ""
    for paragraph in paragraphs:
        paragraphs_content += f"\n {paragraph}"

    prompt = f""" You are given the following query about some aspect of Estonian law: {query}.
    You are also given the following excerpts from the Estonian legal acts: {paragraphs_content}.
    Give the answer to the given query according to the paragraphs provided above. 
    Generalize from them if you are asked about some very specific.
    Answer in a concise but comprehensive way with a very simple language.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["choices"][0]["message"]["content"]
