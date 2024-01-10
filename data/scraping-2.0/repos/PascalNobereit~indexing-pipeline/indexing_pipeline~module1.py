from dotenv import load_dotenv
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager
from langchain.vectorstores import Pinecone
import pinecone


class IndexingPipeline:
    def __init__(self, vectorstore, connector_id: str, db_url: str):
        self.vectorstore = vectorstore
        self.connector_id = connector_id
        self.db_url = db_url

        self.record_manager = None

    def setup(self):
        load_dotenv()
        pinecone.init()


      
        self.record_manager = SQLRecordManager(self.connector_id, db_url=self.db_url)
        self.record_manager.create_schema()

    def run(self, data, source_id_key="source", cleanup="full"):
        from langchain.indexes import index

        return index(
            data,
            self.record_manager,
            self.vectorstore,
            cleanup=cleanup,
            source_id_key=source_id_key,
        )
