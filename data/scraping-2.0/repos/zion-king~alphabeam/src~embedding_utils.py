import os
import time
import openai
import tiktoken
import cohere
import chromadb
import tempfile
import google.generativeai as genai
from llama_index.schema import Document
from llama_index.readers.base import BaseReader
from llama_index.llms import OpenAI, Gemini
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, PromptHelper, LLMPredictor, load_index_from_storage
from llama_index.vector_stores.google.generativeai import GoogleVectorStore, set_google_config, genai_extension as genaix
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer, LLMRerank, CohereRerank, LongContextReorder
from llama_index.embeddings import OpenAIEmbedding, GeminiEmbedding
from llama_index.vector_stores import ChromaVectorStore
from llama_index.memory import ChatMemoryBuffer
from llama_index import download_loader
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from google.oauth2 import service_account


GOOGLE_API_KEY = 'AIzaSyB4Aew8oVjBgPMZlskdhdmQs27DuyNBDAY'
os.environ["GOOGLE_API_KEY"]  = GOOGLE_API_KEY

genai.configure(
    api_key=GOOGLE_API_KEY,
    client_options={"api_endpoint": "generativelanguage.googleapis.com"},
)

CHROMADB_HOST = "localhost"
ALLOWED_EXTENSIONS = {'txt', 'htm', 'html', 'csv', 'yml', 'sql'}


class YMLReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r") as f:
            print(file)
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text + "Foobar", extra_info={"filename": str(file), "file_type": ".yml"})]
        
class SQLReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r") as f:
            print(file)
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text + "Foobar", extra_info={"filename": str(file), "file_type": ".sql"})]


def generate_vector_embedding_chroma(index_name, data_dir):
        
    try:
        # initialize client, setting path to save data
        # db = chromadb.PersistentClient(path="./chroma_db")
        print("Connecting to Chroma database...")
        db = chromadb.HttpClient(host=CHROMADB_HOST, port=8000)
    except:
        return {'statusCode': 400, 'status': 'Could not connect to chroma database'}

    try:
        # create collection
        print("Creating vector embeddings......")
        print("Index name: ", index_name)
        start_time = time.time()
        chroma_collection = db.get_or_create_collection(
            name=index_name,
            metadata={"hnsw:space": "cosine"} # default: L2; used before: ip
            )
    except Exception as e:
        print("Error : : :", e)
        return {'statusCode': 400, 'status': 'A knowledge base with the same name already exists'}

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # setup our storage (vector db)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    llm = Gemini(api_key=GOOGLE_API_KEY, model='models/gemini-pro', temperature=0)
    embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY)
    node_parser = SimpleNodeParser.from_defaults(
        # text_splitter=TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
        chunk_size=1024,
        chunk_overlap=20
        )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
        )

    documents = SimpleDirectoryReader(input_dir=data_dir,
                                      file_extractor={".yml": YMLReader(), ".sql": SQLReader()} # extra custom extractor
                                    ).load_data()
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=service_context
    )

    # data_dir.cleanup() # delete document temp dir

    print(f"Vector embeddings created in {time.time() - start_time} seconds.")

    response = {
        'statusCode': 200,
        'status': 'Chroma embedding complete',
    }
    return response

def generate_vector_embedding_google(index_name, data_dir):

    start_time = time.time()
    vector_store = GoogleVectorStore.create_corpus(corpus_id=index_name) # param:display_name

    llm = Gemini(model='models/gemini-pro', temperature=0)
    embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY)
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=1024,
        chunk_overlap=20
        )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
        )

    # documents = SimpleDirectoryReader(input_dir=data_dir,
    #                                   file_extractor={".yml": YMLReader(), ".sql": SQLReader()} # extra custom extractor
    #                                 ).load_data()
    
    documents = SimpleDirectoryReader(input_dir=data_dir).load_data()

    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        service_context=service_context
    )

    # data_dir.cleanup() # delete document temp dir

    print(f"Vector embeddings created in {time.time() - start_time} seconds.")

    response = {
        'statusCode': 200,
        'status': 'Chroma embedding complete',
    }
    return response


def delete_embeddings():
    return ''













