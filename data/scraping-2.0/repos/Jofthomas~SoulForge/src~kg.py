
import logging
import sys
import time
import subprocess
import os

from llama_index import (
    KnowledgeGraphIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.llms import OpenAI
from llama_index.graph_stores import SimpleGraphStore, NebulaGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index.schema import Document
import openai

from nebula3.gclient.net import Connection
from nebula3.gclient.net.SessionPool import SessionPool
from nebula3.Config import SessionPoolConfig
from nebula3.common.ttypes import ErrorCode

from config import *

openai.api_key = ""
os.environ['NEBULA_USER'] = "root"
os.environ['NEBULA_PASSWORD'] = "nebula"
os.environ["GRAPHD_HOST"] = "127.0.0.1"
os.environ["GRAPHD_PORT"] = "9669"
os.environ['NEBULA_ADDRESS'] = "127.0.0.1:9669"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Knowlege_Graph:
    """Knowlege_Graph class for keeping conversation as knowledge graph format
    """
    def __init__(
        self, 
        llm: OpenAI = OpenAI(temperature=0, model="text-davinci-002"),
        # graph_store = SimpleGraphStore(),
        documents = SimpleDirectoryReader("../data_tmp").load_data()
    ):
        self.llm = llm
        self.graph_store = None
        self.documents = documents
        self.service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)
        # self.storage_context = StorageContext.from_defaults(graph_store=graph_store)

        self.create_graph_space()
        self.connect_nebula_graph()

        self.index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            max_triplets_per_chunk=2,
            service_context=self.service_context,
            include_embeddings=True,
        )

        self.query_engine = self.index.as_query_engine(
            include_text=False, response_mode="tree_summarize"
        )

    def create_graph_space(self):
        config = SessionPoolConfig()

        # prepare space
        conn = Connection()
        conn.open(os.environ["GRAPHD_HOST"], os.environ["GRAPHD_PORT"], 1000)
        auth_result = conn.authenticate(os.environ["NEBULA_USER"], os.environ["NEBULA_PASSWORD"])
        assert auth_result.get_session_id() != 0
        resp = conn.execute(
            auth_result._session_id,
            'CREATE SPACE IF NOT EXISTS SoulForge_test(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);',
        )
        assert resp.error_code == ErrorCode.SUCCEEDED
        # insert data need to sleep after create schema
        time.sleep(10)

        session_pool = SessionPool(os.environ["NEBULA_USER"], os.environ["NEBULA_PASSWORD"], 'SoulForge_test', [(os.environ["GRAPHD_HOST"], os.environ["GRAPHD_PORT"])])
        assert session_pool.init(config)

        # add schema
        resp = session_pool.execute(
            'CREATE TAG IF NOT EXISTS entity(name string);'
            'CREATE EDGE IF NOT EXISTS relationship(relationship string);'
            'CREATE TAG INDEX IF NOT EXISTS entity_index ON entity(name(256));'
        )

    def connect_nebula_graph(self):
        os.environ['NEBULA_USER'] = os.environ["NEBULA_USER"]
        os.environ['NEBULA_PASSWORD'] = os.environ["NEBULA_PASSWORD"]
        os.environ['NEBULA_ADDRESS'] = os.environ["NEBULA_ADDRESS"]

        space_name = "SoulForge_test"
        edge_types, rel_prop_names = ["relationship"], ["relationship"]
        tags = ["entity"]

        graph_store = NebulaGraphStore(
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
        )
        self.storage_context = StorageContext.from_defaults(graph_store=graph_store)

    def query(self, user_question):
        response = self.query_engine.query(user_question)
        #if response=="There is not enough information to answer the query.":
        return response


    def save_conversation(self, chat_history):
        new_data = [Document(text=chat_history)]
        self.index._insert(new_data)



if __name__ == "__main__":
    kg = Knowlege_Graph()
    time.sleep(61)
    res = kg.query("What is K'thrax")
    print(res)
    time.sleep(61)
    res = kg.query("Tell me about America")
    print(res)
    kg.save_conversation("America is a country located in North America.")
    time.sleep(61)
    res = kg.query("Tell me about America")
    print(res)