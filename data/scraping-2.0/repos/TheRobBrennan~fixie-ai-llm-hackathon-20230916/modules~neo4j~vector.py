from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings


def initialize_neo4j_vector(credentials, index_name):
    """
    Function to instantiate a Neo4j vector from an existing vector.
    """
    # Implement the actual logic using the langchain and neo4j modules here
    # Neo4j Aura credentials
    url = credentials["url"]
    username = credentials["username"]
    password = credentials["password"]

    # OpenAI credentials
    openai_api_secret_key = credentials["openai_api_secret_key"]

    # Instantiate Neo4j vector from an existing vector
    # CYPHER - "SHOW INDEXES;" will show we have an index type Vector named "vector"
    neo4j_vector = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(openai_api_key=openai_api_secret_key),
        url=url,
        username=username,
        password=password,
        index_name=index_name,
    )

    return neo4j_vector


def perform_similarity_search(neo4j_vector, query):
    """
    Function to perform a vector similarity search.
    """
    # Implement the actual logic using the langchain module's similarity_search method
    try:
        results = neo4j_vector.similarity_search(query)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return results


def store_data_in_neo4j(documents, credentials):
    """
    Store and index text with Neo4j.
    """
    # Neo4j Aura credentials
    url = credentials["url"]
    username = credentials["username"]
    password = credentials["password"]

    # OpenAI credentials
    openai_api_secret_key = credentials["openai_api_secret_key"]

    # Instantiate Neo4j vector from documents
    Neo4jVector.from_documents(
        documents,
        OpenAIEmbeddings(openai_api_key=openai_api_secret_key),
        url=url,
        username=username,
        password=password,
    )
