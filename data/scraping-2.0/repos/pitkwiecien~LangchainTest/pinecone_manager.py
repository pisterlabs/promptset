# noinspection PyPackageRequirements
import pinecone
# noinspection PyPackageRequirements
from pinecone.core.client.exceptions import NotFoundException, ApiException
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

import config
from file_manager import FileManager


class PineconeManager:
    __instance = None

    def __init__(self, index_name):
        self.index_name = index_name
        pinecone.init(environment='gcp-starter')
        try:
            _ = pinecone.describe_index(index_name)
            
        except NotFoundException:
            try:
                pinecone.create_index(index_name, 1536)
                
            except ApiException:
                indices = pinecone.list_indexes()
                for index in indices:
                    pinecone.delete_index(index)
                pinecone.create_index(index_name, 1536)

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(PineconeManager, cls).__new__(cls)
        return cls.__instance

    def index_content(self, content):
        embeddings_model = OpenAIEmbeddings()
        index = pinecone.Index(self.index_name)
        index.delete(delete_all=True)
        Pinecone.from_documents(
            FileManager.content_for_pinecone(content),
            embeddings_model,
            index_name=self.index_name
        )

    def get_index(self):
        embeddings_model = OpenAIEmbeddings()
        return Pinecone.from_existing_index(self.index_name, embeddings_model)
