import os

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

load_dotenv()

url: str = os.environ.get("SUPABASE_URL", "default_url")
key: str = os.environ.get("SUPABASE_KEY", "default_key")
supabaseClient: Client = create_client(url, key)
##response = supabase.table("coins_details").select("*").execute()
embeddings = OpenAIEmbeddings()
vector_store = SupabaseVectorStore(embedding=embeddings, client=supabaseClient, table_name="documents", query_name="match_documents")

# create a getter function for the vector store
def get_vector_store_retriever() :
    return vector_store.as_retriever()

def saveToSupabase(content: str, metadata: dict, embedding: list):
    response = supabaseClient.table("documents").upsert({"content": content, "metadata": metadata, "embedding": embedding}).execute()
    
def saveDocVectorSupabase(docs: list):
     supabaseVec=vector_store.from_documents(docs, embeddings, client=supabaseClient, table_name="documents")

     
def getSimilarDocuments(text: str):
      return vector_store.similarity_search(text, 10)
     

    
     
     
# def getSimilarDocuments(text: str, embeddings: Embeddings):
#     return SupabaseVectorStore.similarity_search(query=text,)
    
    
 


