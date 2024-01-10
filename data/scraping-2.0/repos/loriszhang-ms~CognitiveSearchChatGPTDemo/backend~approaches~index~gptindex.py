from approaches.index.store.cosmos_index_store import CosmosIndexStore
from llama_index import StorageContext
from approaches.index.store.cosmos_doc_store import CosmosDocumentStore
from llama_index import load_index_from_storage
import os
import openai
from langchain.chat_models  import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    LLMPredictor,
    ServiceContext
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index import SimpleDirectoryReader, Document
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from dotenv import load_dotenv

load_dotenv()

AZURE_INDEX_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_INDEX_STORAGE_CONNECTION_STRING")
AZURE_OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_BASE")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY_SOUTH_CENTRAL_US")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_API_BASE
openai.api_version = "2023-03-15-preview"
os.environ["OPENAI_API_KEY"] = str(AZURE_OPENAI_API_KEY)
openai.api_key = AZURE_OPENAI_API_KEY

class GPTKGIndexer:

    def __init__(self):
        self._connection_string = AZURE_INDEX_STORAGE_CONNECTION_STRING
        self._index_store = CosmosIndexStore.from_uri(uri=str(self._connection_string), db_name="kg_index")
        self._doc_store = CosmosDocumentStore.from_uri(uri=str(self._connection_string), db_name = "doc_store")
        self._storage_context = StorageContext.from_defaults(
            docstore=self._doc_store,
            index_store=self._index_store)
        self._llm = AzureChatOpenAI(deployment_name="gpt-4", 
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base,
            openai_api_type=openai.api_type,
            openai_api_version=openai.api_version,
            temperature=0.0
        )
        llm_predictor = LLMPredictor(llm=self._llm)

        self._embedding_llm = LangchainEmbedding(
            OpenAIEmbeddings(
                model="text-embedding-ada-002",
                deployment="text-embedding-ada-002",
                openai_api_key= openai.api_key,
                openai_api_base=openai.api_base,
                openai_api_type=openai.api_type,
                openai_api_version=openai.api_version,
            ),
            embed_batch_size=1,
        )
        self._service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024)
        try:
            print("Loading index from storage")
            self.index = load_index_from_storage(storage_context=self._storage_context, service_context = self._service_context)
            print("Index loaded from storage")
        except:
            print("Initializing new index")
            self.index = self._init_index()
            print("Initialized new index")
        

    def add_document(self, fileContent: str):
        text_splitter = TokenTextSplitter(separator=" ", chunk_size=2048, chunk_overlap=20)
        text_chunks = text_splitter.split_text(fileContent)
        doc_chunks = [Document(t) for t in text_chunks]
        for doc_chunk in doc_chunks:
            self.index.insert(doc_chunk)

    def query(self, question: str):
        query_engine = self.index.as_query_engine(
            include_text=False, 
            response_mode="tree_summarize"
        )
        response = query_engine.query(question)
        return response

    def _init_index(self):
        self.index = GPTKnowledgeGraphIndex(
            [],
            service_context=self._service_context,
            storage_context=self._storage_context
        )