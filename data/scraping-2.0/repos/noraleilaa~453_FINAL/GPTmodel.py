import sys
import openai
import key
import llama_index
import wikipedia
from llama_index import (
    VectorStoreIndex,
    get_response_synthesizer,
    Document,
    SimpleDirectoryReader,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor

import langchain
import os

openai.api_key = key.API_KEY


def download_wikipedia_article(page_title):
    try:
        page = wikipedia.page(page_title)
        return page.content
    except wikipedia.exceptions.PageError as e:
        return f"Page not found: {e}"
    

# Example usage
documents = download_wikipedia_article("Greek mythology")
print(documents[:500])  # Print first 500 characters to check


documents = SimpleDirectoryReader('./data').load_data()
# Create an index of your documents


index = VectorStoreIndex(documents)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# query
response1 = query_engine.query("What is Greek mythology?")
print(response1)

response2 = query_engine.query('Who is Zues?')
print(response2)

response3 = query_engine.query("What group of individuals have derived inspiration from Greek Mythology?")
print(response3)

response4 = query_engine.query("Who was Hesiod to Homer?")
print(response4)

response5 = query_engine.query("Why has Greek mythology changed over time?")
print(response5)
