from serpapi import GoogleSearch
import os

import requests
from bs4 import BeautifulSoup
import re

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
import os
from deeplake.core.vectorstore import VectorStore

import openai


def search_news():
    search = GoogleSearch({
        "engine": "google",
        "q": "Berita kesehatan health terkini penyakit",
        "location_requested": "Indonesia",
        "location_used": "Indonesia",
        "google_domain": "google.co.id",
        "hl": "id",
        "gl": "id",
        "device": "desktop",
        "tbm": "nws",
        "num": "10",
        "api_key": os.getenv("SERP_API_KEY")
    })

    result = search.get_dict()

    return result


def embedding_function(texts, model="text-embedding-ada-002"):
    if isinstance(texts, str):
        texts = [texts]

    texts = [t.replace("\n", " ") for t in texts]
    return [data['embedding']for data in openai.Embedding.create(input=texts, model=model)['data']]


def get_news_content():
    res = search_news()

    links = [x["link"] for x in res["news_results"]]

    contents = dict()
    for l in links:
        r = requests.get(l)

        if r.status_code != 200:
            continue

        soup = BeautifulSoup(r.content, 'html.parser')
        text = re.sub(r'\s+', ' ', soup.text.replace("\n", " "))

        contents[l] = text

    return contents


def store_and_embed_news(contents, chunk_size=1000):
    dataset_path = 'hub://luisfrentzen/data'
    vector_store = VectorStore(
        path=dataset_path,
    )

    for s, c in contents.items():
        chunked_text = [c[i:i+1000] for i in range(0, len(c), chunk_size)]

        print(s)
        vector_store.add(text=chunked_text,
                         embedding_function=embedding_function,
                         embedding_data=chunked_text,
                         metadata=[{"source": s}]*len(chunked_text))
