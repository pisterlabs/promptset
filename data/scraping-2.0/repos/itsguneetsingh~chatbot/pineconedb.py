from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
import constants
import pinecone

embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index = pinecone.Index(constants.PINECONE_INDEX_NAME)
vectorstore = Pinecone(index, embed_model, "text")

def search(query):
    query_vector = embed_model.encode(query)
    
    results = index.query(query_vector, top_k=2)
    
    return [result.id for result in results]