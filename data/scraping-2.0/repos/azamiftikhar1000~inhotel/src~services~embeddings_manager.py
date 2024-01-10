import os
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = None


# It will be useful for loading opensource models
def setup_embedding_model():
    """Setup embedding model"""
    global embeddings_model
    if embeddings_model is None:
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    return embeddings_model
