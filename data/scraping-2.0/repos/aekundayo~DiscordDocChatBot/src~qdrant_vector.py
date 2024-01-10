from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Qdrant, weaviate, Redis
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
  
from qdrant_client.http.models import CollectionDescription
import os

vectorpath = 'docs/vectorstore'
url = os.getenv("QDRANT_HOST_STRING"),


def get_Qvector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    #encoder = SentenceTransformer("all-MiniLM-L6-v2")

    memory_vectorstore = Qdrant.from_texts(
    text_chunks,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="current_document",
    force_recreate=True,
    )


    try: 
        client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST_STRING")
        )
        qdrant = Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name="my_documents"
        )
        qdrant.add_texts(text_chunks)
    except:
        try:
            qdrant = qdrant_client.QdrantClient(url=os.getenv("QDRANT_HOST_STRING"))
            collection_created = qdrant.recreate_collection(
            collection_name="my_documents",
            vectors_config=models.VectorParams(
                size=1536,  # Vector size is defined by used model
                distance=models.Distance.COSINE,
            ),
        )
            print("VECTOR DB CREATED RESULT: " + collection_created)

            qdrant.add_texts(text_chunks)       

        except:
            pass


    return memory_vectorstore 

def get_Qvector_store_from_docs(documents):
    embeddings = OpenAIEmbeddings()
    #encoder = SentenceTransformer("all-MiniLM-L6-v2")
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    
    memory_vectorstore = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="current_document",
    force_recreate=True,
    )

    try: 
        client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST_STRING")
        )
        qdrant = Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name="my_documents"
        )
        qdrant.add_documents(documents)
    except:
        try:
            qdrant = qdrant_client.QdrantClient(url=os.getenv("QDRANT_HOST_STRING"))
            collection_created= qdrant.recreate_collection(
            collection_name="my_documents",
            vectors_config=models.VectorParams(
                size=1536,  # Vector size is defined by used model
                distance=models.Distance.COSINE,
            ),
        )
            print("VECTOR DB CREATED RESULT: " + collection_created)
            qdrant.add_documents(documents)       

        except:
            pass


    return memory_vectorstore 



def return_qdrant():    
    embeddings = OpenAIEmbeddings()
    client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST_STRING")
    )
    qdrant = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="my_documents"
    )

    return qdrant

def update_qdrant(text_chunks):
    embeddings = OpenAIEmbeddings()
    client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST_STRING")
    )
    qdrant = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="my_documents"
    )
    qdrant.add_texts(text_chunks)



def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0



