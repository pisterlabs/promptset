import weaviate
from llama_index.vector_stores import WeaviateVectorStore
from llama_index import VectorStoreIndex, StorageContext
from .config import OPENAI_API_KEY

def setup_weaviate_client():
    """
    Initializes and returns a Weaviate client with the specified configurations.

    Returns:
    - A Weaviate client instance.
    """
    
    # Set up a Weaviate client with OpenAI API key header
    return weaviate.Client(embedded_options=weaviate.embedded.EmbeddedOptions(),
                           additional_headers={'X-OpenAI-Api-Key': OPENAI_API_KEY})

def setup_vector_index(client, nodes):
    """
    Sets up and returns a vector index using the provided Weaviate client and data nodes.

    Args:
    - An initialized Weaviate client instance.
    - nodes (list): A list of data nodes to index.

    Returns:
    - A VectorStoreIndex instance with the indexed nodes (llama_index).
    """

    # Set up vector store using the provided Weaviate client
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="PaperText", text_key="content")

    # Initialize storage context with the vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create and return a VectorStoreIndex with the provided nodes and storage context
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index
