import logging

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from vector_storage.azure_vector_storage import AzureVectorStorage


class ChatBotController:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatBotController, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not self.__initialized:
            self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            self.index_name = "genaipartners"
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.vector_storage = AzureVectorStorage(self.index_name, self.embeddings)

            self.__initialized = True

    def initialize(self):
        logging.info("Dummy function to load embeddings when server starts")

    def query(self, query_str):
        _, query_str = query_str.split(":")
        docs = self.vector_storage.search(query_str)
        return docs[0].page_content

    async def add_document(self, document, content_type):
        logging.info(f"Adding {document}")

        if content_type == "application/pdf":
            loader = PyPDFLoader(document)
            documents = loader.load_and_split()

        docs = self.text_splitter.split_documents(documents)
        ids = await self.vector_storage.add_documents(docs)
        logging.info(f"Document added {ids}")
