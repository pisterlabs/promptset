import os
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import OpenAIEmbeddings

# Retrieve environment variables for sensitive information
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

ATLAS_CONNECTION_STRING = os.getenv('ATLAS_CONNECTION_STRING')
if not ATLAS_CONNECTION_STRING:
    raise ValueError("The ATLAS_CONNECTION_STRING environment variable is not set.")

# Set the OPENAI_API_KEY in the environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DB_NAME = "langchain"
COLLECTION_NAME = "vectorSearch"

def create_vector_search():
    """
    Creates a MongoDBAtlasVectorSearch object using the connection string, database, and collection names, along with the OpenAI embeddings and index configuration.
    """
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        ATLAS_CONNECTION_STRING,
        f"{DB_NAME}.{COLLECTION_NAME}",
        OpenAIEmbeddings(),
        index_name="default"
    )
    return vector_search

def perform_similarity_search(query, top_k=3):
    """
    This function performs a similarity search within a MongoDB Atlas collection using the MongoDBAtlasVectorSearch object.
    """
    vector_search = create_vector_search()
    
    # Execute the similarity search with the given query
    results = vector_search.similarity_search_with_score(
        query=query,
        k=top_k,
    )
    
    return results

# Example usage of the perform_similarity_search function
if __name__ == "__main__":
    try:
        query = "How does MongoDB Atlas handle security?"
        top_k_results = 3
        search_results = perform_similarity_search(query, top_k=top_k_results)
        
        print(f"Top {top_k_results} results for the query '{query}':")
        for result in search_results:
            print(result)
    except Exception as e:
        print(f"An error occurred: {e}")
