import os
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index import Document

from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
load_dotenv()
# initialize connection to pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)

# create the index if it does not exist already


def create_index(index_name):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )
    else:
        print("Index already exists. Skipping index creation.")

def embed(index_name: str, docs: list, model: str='text-embedding-ada-002', embed_batch_size: int=100):
    # create the index if it does not exist already
    create_index(index_name)
    # connect to the index
    pinecone_index = pinecone.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # setup our storage (vector db)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    # setup the index/query process, ie the embedding model (and completion if used)
    embed_model = OpenAIEmbedding(model=model, embed_batch_size=embed_batch_size)

    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    index = GPTVectorStoreIndex.from_documents(
        preprocess_doc(docs), storage_context=storage_context,
        service_context=service_context
    )
    
def preprocess_doc(document):
    docs = []
    for i, row in enumerate(document):
        docs.append(Document(
            text=row.page_content,
            doc_id=int(i),
        ))
    return docs