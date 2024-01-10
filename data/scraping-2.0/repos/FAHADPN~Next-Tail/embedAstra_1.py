import os
from dotenv import load_dotenv
# from astrapy.db import AstraDB
import cassio
# LangChain components to use
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import CohereEmbeddings
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.vector_stores import  AstraDBVectorStore
from langchain.vectorstores import AstraDB


from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)


load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# cohereEmbedModel = CohereEmbeddings(
#     model="embed-english-v3.0",
#     cohere_api_key= os.getenv("COHERE_API_KEY")
# )
# cohereEmbedModel = CohereEmbedding(
#     cohere_api_key=COHERE_API_KEY,
#     model_name="embed-english-v3.0",
#     input_type="search_query",
# )

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)


def embedIntoAstra(docs, VectorEmbedding):
    """
    Integerate Langchain and astra DB to create embedding using Cohere embed model and
    add to Astra DB
    """
    cohereEmbedModel = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key= os.getenv("COHERE_API_KEY")
    )

    """Langchain Vector store backed by Astra"""
    print("ENTEERING ASTRA")

    astra_vector_store = AstraDB(
        embedding=cohereEmbedModel,
        collection_name="astra_vector_demo_ron",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )

    print(astra_vector_store)

    if (astra_vector_store):
        print("VECTOR DB CONNECTION ESTABLISHED")
    else :
        print("ASTRA FAILED")

    inserted_ids = astra_vector_store.add_documents(docs)
    print(f"\nInserted {len(inserted_ids)} documents.")
    # astra_vector_store.from_documents(documents=documents ,embedding=VectorEmbedding)
    # astra_vector_store.add_texts([documents])
    # # print(astra_vector_store.load_data())

    # astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    # print(astra_vector_index)


def embedIntoAstraLLAMA_INDEX(documents):
    astra_vector_store = AstraDBVectorStore(
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name="tailwindDocs",
        embedding_dimension=1536,
    )

    # storage_context = StorageContext.from_defaults(vector_store=astra_vector_store)

    # index = VectorStoreIndex.from_documents(
    #     documents, storage_context=storage_context
    # )
    # print(len(index))
    
    astra_vector_store.add_texts(documents)
    print(astra_vector_store.load_data())

    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    print(astra_vector_index)
        



# def main():
#     ...

# if __name__ == "__main__":
#     embedIntoAstra()

