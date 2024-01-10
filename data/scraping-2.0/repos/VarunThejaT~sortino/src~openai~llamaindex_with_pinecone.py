import logging
import sys
import openai
import os
from dotenv import load_dotenv

from llama_index import (
    GPTVectorStoreIndex,
    GPTSimpleKeywordTableIndex, 
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
from llama_index.vector_stores import PineconeVectorStore
from langchain.llms.openai import OpenAIChat

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


wiki_titles = ["Boston", "Chicago"]
pinecone_titles = ["boston", "chicago"]

from pathlib import Path

import requests
for title in wiki_titles:
    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            # 'exintro': True,
            'explaintext': True,
        }
    ).json()
    page = next(iter(response['query']['pages'].values()))
    wiki_text = page['extract']

    data_path = Path('data')
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", 'w') as fp:
        fp.write(wiki_text)

city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(input_files=[f"data/{wiki_title}.txt"]).load_data()

import pinecone
api_key = os.getenv("PINECONE_API_KEY")
environment = "northamerica-northeast1-gcp"
index_name = "openai"

# LLM Predictor (gpt-3.5-turbo)
llm_predictor_chatgpt = LLMPredictor(
    llm=OpenAIChat(temperature=0, model_name="gpt-3.5-turbo")
)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt)

# Build city document index
from llama_index.storage.storage_context import StorageContext


city_indices = {}
for pinecone_title, wiki_title in zip(pinecone_titles, wiki_titles):
    metadata_filters = {"wiki_title": wiki_title}
    vector_store = PineconeVectorStore(
        index_name=index_name,
        environment=environment,
        metadata_filters=metadata_filters,
        namespace="cities" # additional setting to group the embeddings
    ) 
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    city_indices[wiki_title] = GPTVectorStoreIndex.from_documents(
        city_docs[wiki_title], storage_context=storage_context, service_context=service_context
    )
    # set summary text for city
    city_indices[wiki_title].index_struct.index_id = pinecone_title


response = city_indices["Boston"].as_query_engine(service_context=service_context).query(
    "Tell me about the arts and culture of Boston"
)

print(str(response))
print(response.get_formatted_sources())


from llama_index.indices.composability.graph import ComposableGraph

# set summaries for each city
index_summaries = {}
for wiki_title in wiki_titles:
    # set summary text for city
    index_summaries[wiki_title] = f"Wikipedia articles about {wiki_title}"

graph = ComposableGraph.from_indices(
    GPTSimpleKeywordTableIndex,
    [index for _, index in city_indices.items()], 
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50
)
custom_query_engines = {
    graph.root_id: graph.root_index.as_query_engine(retriever_mode='simple', service_context=service_context)
}

query_engine = graph.as_query_engine(
    custom_query_engines=custom_query_engines,
)
