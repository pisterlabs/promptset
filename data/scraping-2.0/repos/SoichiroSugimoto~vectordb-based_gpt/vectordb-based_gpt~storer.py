import os
import sys
import json
import logging
import inspect
import pprint
from datetime import datetime
from dynamodb_client import DynamoDBTable
from dotenv import load_dotenv

from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    StringIterableReader,
    LLMPredictor,
    ServiceContext,
)
from langchain.llms.openai import OpenAIChat
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import PineconeVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
load_dotenv('.env')

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class IndexCreator:
    def __init__(self, pinecone_index_name=None, pinecone_environment=None, model_name=None):
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_environment = pinecone_environment
        self.llm = OpenAIChat(temperature=0, model=model_name)
        self.service_context = ServiceContext.from_defaults(llm=self.llm)

    # ベクトルデータの保存先を作成
    def _create_vector_store(self, accessibility_id="001"):
        return PineconeVectorStore(
            index_name=self.pinecone_index_name,
            environment=self.pinecone_environment,
            namespace=accessibility_id
        )
    
    # ベクトルデータを作成、保存
    def _create_vector_store_index(self, article, vector_store):
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(
            article,
            storage_context=storage_context,
            service_context=self.service_context,
        )

    def insert_to_pinecone(self, article, accessibility_id="001"):
        vector_store = self._create_vector_store(accessibility_id)
        vector_store_index = self._create_vector_store_index(article, vector_store)
        nodes = vars(vector_store_index)['_nodes']
        return nodes


class DynamoDBInserter:
    def __init__(self, table_name, partition_key_name, sort_key_name):
        self.dynamodb = DynamoDBTable(
            table_name=table_name,
            region_name="ap-northeast-1",
            partition_key_name=partition_key_name,
            sort_key_name=sort_key_name,
        )

    def _create_item(self, pinecone_id, article_summary, category_id):
        return {
            "pinecone_id": pinecone_id,
            "summary": article_summary,
            "category_id": category_id,
            "deleted": 0,
            "created_by": inspect.currentframe().f_code.co_name,
            "updated_by": inspect.currentframe().f_code.co_name,
        }

    def insert_to_dynamodb(self, pinecone_id, article_summary, category_id):
        item = self._create_item(pinecone_id, article_summary, category_id)
        self.dynamodb.put_item(item)


def create_index(article_summary, article, accessibility_id):
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    model_name = "text-embedding-ada-002"
    index_creator = IndexCreator(pinecone_index_name, pinecone_environment, model_name)
    pinecone_nodes = index_creator.insert_to_pinecone(article, accessibility_id)
    dynamodb_inserter = DynamoDBInserter("Article", "category_id", "deleted")

    for i, node in enumerate(pinecone_nodes, start=1):
        now = datetime.now()
        date_string = now.strftime("%Y%m%d%H%M%S")
        pinecone_id = vars(node)["id_"]
        category_id = f"{accessibility_id}#{date_string}#{i:03}"
        dynamodb_inserter.insert_to_dynamodb(pinecone_id, article_summary, category_id)


def create_index_from_string(article_summary, text, accessibility_id):
    article = StringIterableReader().load_data(texts=[text])
    create_index(article_summary, article, accessibility_id)
