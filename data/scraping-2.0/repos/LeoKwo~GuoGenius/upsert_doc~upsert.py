from pathlib import Path
from llama_hub.file.unstructured import UnstructuredReader
from pathlib import Path
from llama_index import download_loader
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from dotenv import load_dotenv
import os
from llama_index.node_parser import SimpleNodeParser
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
import openai

####################################################
#                                                  #
# This file upserts documents in data to pinecone. #
#                                                  #
####################################################

load_dotenv()
openai.api_key = os.getenv('api_key')
# find API key in console at app.pinecone.io
os.environ['PINECONE_API_KEY'] = os.getenv('pinecone_api_key')
# environment is found next to API key in the console
os.environ['PINECONE_ENVIRONMENT'] = os.getenv('pinecone_env')

# loader = UnstructuredReader()

# initialize connection to pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)

# setup the index/query process, ie the embedding model (and completion if used)
embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# Readers
PDFReader = download_loader("PDFReader")
MarkdownReader = download_loader("MarkdownReader")

# Load docs
def upsert_docs(input_dir: str, index_name: str):
    print(f"Building from {input_dir} under index {index_name}...\n")
    documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
    
    # create the index if it does not exist already
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine'
        )

    # connect to the index
    pineconeIndex = pinecone.Index(index_name)

    vectorStore = PineconeVectorStore(
        pinecone_index=pineconeIndex
    )

    # setup our storage (vector db)
    storageContext = StorageContext.from_defaults(
        vector_store=vectorStore
    )

    index = GPTVectorStoreIndex.from_documents(
        documents=documents, 
        storage_context=storageContext,
        service_context=service_context
    )
    print(f"Done building !\n")

upsert_docs(input_dir="upsert_doc/docs", index_name="ruikang-guo-knowledge-base")
