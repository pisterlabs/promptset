import os
from typing import List, Literal, Optional
import openai
import qdrant_client
import weaviate
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from llama_index import Document, LangchainEmbedding, OpenAIEmbedding, Prompt, QueryBundle, SimpleDirectoryReader, SimpleWebPageReader, StorageContext, VectorStoreIndex, download_loader, load_index_from_storage
from llama_index.llms import AzureOpenAI, OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.graph_stores import Neo4jGraphStore
from llama_index.indices.base import BaseIndex
from llama_index.schema import NodeWithScore
from llama_index.indices.postprocessor.node import BasePydanticNodePostprocessor


## ----------------------------------------
## ■ Embedding Model
## ----------------------------------------
def embed_azure() -> LangchainEmbedding:
  """
  AOAI Embedding Model
  :return: LangchainEmbedding
  """
  os.environ["OPENAI_API_KEY"]     = os.environ["AOAI_API_KEY"]
  os.environ["OPENAI_API_BASE"]    = os.environ["AOAI_API_HOST"]
  os.environ["OPENAI_API_TYPE"]    = "azure"
  os.environ["OPENAI_API_VERSION"] = "2023-05-15"

  return LangchainEmbedding(OpenAIEmbeddings(model="text-embedding-ada-002",deployment="text-embedding-ada-002_base"),embed_batch_size=1)

def embed_openai() -> OpenAIEmbedding:
  """
  OpenAI Embedding Model
  :return: OpenAIEmbedding
  """
  return OpenAIEmbedding(model="text-embedding-ada-002")

def embed_langchain(model:Literal[
    "intfloat/e5-large-v2",
    "intfloat/e5-base-v2",
    "intfloat/multilingual-e5-large",
    "intfloat/e5-large"
  ]) -> LangchainEmbedding:
  """
  Langchain Embedding Model
  :param model: model name
  :return: LangchainEmbedding
  """
  return LangchainEmbedding(HuggingFaceEmbeddings(model_name=model))


## ----------------------------------------
## ■ LLM Model
## ----------------------------------------
def llm_azure() -> AzureOpenAI:
  """
  AOAI LLM Model
    -> model : text-davinci-003 | gpt-35-turbo | gpt-35-turbo-16k | gpt-4 | gpt-4-32k
    -> engine: text-davinci-003_base | gpt-35-turbo_base | gpt-35-turbo-16k_base | gpt-4_base | gpt-4-32k_base
  """
  openai.api_key     = os.environ["AOAI_API_KEY"]
  openai.api_base    = os.environ["AOAI_API_HOST"]
  openai.api_type    = "azure"
  openai.api_version = "2023-05-15"

  return AzureOpenAI(model="gpt-4", engine="gpt-4_base", temperature=0, max_tokens=800)

def llm_openai() -> OpenAI:
  """
  OpenAI LLM Model
    -> model : text-davinci-003 | gpt-3.5-turbo
  """
  return OpenAI(model="gpt-3.5-turbo")


## ----------------------------------------
## ■ Load Documents
## ----------------------------------------
def load_documents_local_files(dir_path:str) -> list[Document]:
  """
  ローカル環境のファイルをドキュメントとして読み込む
  https://gpt-index.readthedocs.io/en/latest/examples/data_connectors/simple_directory_reader.html
    -> Usage:
      -> documents = load_documents.load_documents_local_files("../data")
  :param dir_path: directory path | e.g. "../data"
  """
  DocxReader         = download_loader("DocxReader")
  JSONReader         = download_loader("JSONReader")
  PagedCSVReader     = download_loader("PagedCSVReader")
  PDFMinerReader     = download_loader("PDFMinerReader")
  UnstructuredReader = download_loader('UnstructuredReader')
  return SimpleDirectoryReader(dir_path, file_extractor={
    ".csv": PagedCSVReader(),
    ".docx": DocxReader(),
    ".html": UnstructuredReader(),
    ".json": JSONReader(),
    ".pdf": PDFMinerReader(),
  }, recursive=True, required_exts=[".csv",".docx",".html",".json",".pdf"]).load_data()

def load_documents_web_page(web_pages:list[str]) -> list[Document]:
  """
  Webページをドキュメントとして読み込む
  https://gpt-index.readthedocs.io/en/latest/examples/data_connectors/WebPageDemo.html
    -> Usage:
      -> documents = load_documents.load_documents_web_page(["https://ja.wikipedia.org/wiki/ONE_PIECE"])
  :param web_pages: list of web pages | e.g. ["http://paulgraham.com/worked.html"]
  """
  return SimpleWebPageReader(html_to_text=True).load_data(web_pages)


## ----------------------------------------
## ■ Load Index
## ----------------------------------------
def load_index_graph_store_knowledge():
  storage_context = StorageContext.from_defaults(persist_dir="../storages/graph_store/knowledge")
  return load_index_from_storage(storage_context=storage_context)

def load_index_graph_store_neo4j():
  graph_store = Neo4jGraphStore(username="neo4j",password="Admin-999",url="bolt://neo4j:7687",database="neo4j")
  storage_context = StorageContext.from_defaults(graph_store=graph_store, persist_dir="../storages/graph_store/neo4j")
  return load_index_from_storage(storage_context=storage_context)

def load_index_list_store_simple():
  storage_context = StorageContext.from_defaults(persist_dir="../storages/list_index/simple")
  return load_index_from_storage(storage_context=storage_context)

def load_index_vector_store_faiss():
  dir_path = "../storages/vector_store/faiss"
  vector_store = FaissVectorStore.from_persist_dir(dir_path)
  storage_context = StorageContext.from_defaults(vector_store=vector_store,persist_dir=dir_path)
  return load_index_from_storage(storage_context=storage_context)

def load_index_vector_store_qdrant():
  dir_path = "../storages/vector_store/qdrant"
  client = qdrant_client.QdrantClient(path=dir_path)
  vector_store = QdrantVectorStore(client=client, collection_name="LlamaIndex")
  storage_context = StorageContext.from_defaults(vector_store=vector_store,persist_dir=dir_path)
  return load_index_from_storage(storage_context=storage_context)

def load_index_vector_store_simple():
  storage_context = StorageContext.from_defaults(persist_dir="../storages/vector_store/simple")
  return load_index_from_storage(storage_context=storage_context)

def load_index_vector_store_weaviate():
  client = weaviate.Client("http://weaviate:8080")
  vector_store = WeaviateVectorStore(weaviate_client=client, index_name="LlamaIndex")
  return VectorStoreIndex.from_vector_store(vector_store)


## ----------------------------------------
## ■ Load Query Engine
## ----------------------------------------
def load_query_engine_for_knowledge_graph(index:BaseIndex):
  return index.as_query_engine(
    include_text   = True,
    response_mode  = "tree_summarize",
    embedding_mode = "hybrid",
    verbose        = True
  )

def load_query_engine_for_simple(index:BaseIndex, stream_mode:bool=False):
  return index.as_query_engine(streaming=stream_mode, verbose=True)


## ----------------------------------------
## ■ Custom Prompt
## ----------------------------------------
def custom_prompt_condense_question_prompt():
  """
  Chat Engine Custom Prompt
    -> 要約質問プロンプト
  """
  return Prompt("""\
    Given a conversation (between Human and Assistant) and a follow up message from Human, \
    rewrite the message to be a standalone question that captures all relevant context \
    from the conversation.

    <Chat History>
    {chat_history}

    <Follow Up Message>
    {question}

    <Standalone question>
  """)


## ----------------------------------------
## ■ Custom Node Processor
## ----------------------------------------
class CustomNodePostprocessor(BasePydanticNodePostprocessor):

  nodes: List[NodeWithScore] = []

  def postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]) -> List[NodeWithScore]:
    return self.nodes