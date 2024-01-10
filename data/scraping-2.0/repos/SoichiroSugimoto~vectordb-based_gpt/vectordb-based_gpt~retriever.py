import os
import sys
import logging
import pinecone_client as pinecone
import openai
from dotenv import load_dotenv
from dynamodb_client import DynamoDBTable

from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    StringIterableReader,
    LLMPredictor,
    ServiceContext,
)
from llama_index.indices.composability.graph import ComposableGraph
from llama_index.indices.vector_store import GPTVectorStoreIndex
from langchain.llms.openai import OpenAIChat
from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
load_dotenv('.env')

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


def create_index_from_pinecone(namespace):
    pinecone_instance = pinecone.PineconeClient(
        os.getenv("PINECONE_API_KEY"),
        os.getenv("PINECONE_ENVIRONMENT"),
        os.getenv("PINECONE_INDEX_NAME"))
    vector_store = PineconeVectorStore(pinecone_instance.index, namespace=namespace)
    index = GPTVectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


def setup_retriever():
    documents = StringIterableReader().load_data("exec not to store docs, but to setup retriever")
    llm = OpenAI(temperature=0, model="gpt-4")
    service_context = ServiceContext.from_defaults(llm=llm)
    VectorStoreIndex.from_documents(
            documents=documents,
            llm_predictor=LLMPredictor(),
            service_context=service_context
        )
    return None


def create_query_engine(accessibility_ids):
    setup_retriever()
    if accessibility_ids is None:
        index = create_index_from_pinecone(namespace="")
    else:
        index = create_index_from_pinecone(namespace=accessibility_ids[0])
    query_engine = index.as_query_engine()
    return query_engine


""""
def get_pinecone_ids_from_dynamo_db(accessibility_ids):
    ids = []
    article_table = DynamoDBTable(
        table_name="Article", region_name="ap-northeast-1", partition_key_name="category_id", sort_key_name="deleted"
    )
    if accessibility_ids is None:
        records = article_table.get_alive_records()
    else:
        records = article_table.query_items(partition_key_prefixes=accessibility_ids, sort_key_value=0)
    if records is not None:
        for record in records:
            ids.append(record["pinecone_id"])
    return ids


def get_vector_store_index_summary_sets_from_dynamo_db():
    indicies = []
    summaries = []
    data_set = {}
    article_table = DynamoDBTable(
        table_name="Article", region_name="ap-northeast-1", partition_key_name="category_id", sort_key_name="deleted"
    )
    records = article_table.query_items(partition_key_prefixes=["001"], sort_key_value=0)
    if records is not None:
        for record in records:
            if not record["summary"] in data_set:
                data_set[record["summary"]] = [record["pinecone_id"]]
            else:
                data_set[record["summary"]].append(record["pinecone_id"])
        for summary in data_set:
            indicies.append(get_index_from_pinecone_ids(data_set[summary]))
            summaries.append(summary)
    return {"indicies": indicies, "summaries": summaries}


def build_composable_graph():
    index_summary_sets = get_vector_store_index_summary_sets_from_dynamo_db()
    indices = index_summary_sets["indicies"]
    summaries = index_summary_sets["summaries"]
    graph = ComposableGraph.from_indices(
        SimpleKeywordTableIndex,
        indices,
        summaries,
        max_keywords_per_chunk=50,
    )
    return graph


def create_query_engine():
    graph = build_composable_graph()
    llm = OpenAIChat(temperature=0, model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)
    custom_query_engines = {
        graph.root_id: graph.root_index.as_query_engine(retriever_mode="simple", service_context=service_context)
    }
    query_engine = graph.as_query_engine(
        custom_query_engines=custom_query_engines,
    )
    return query_engine
"""
