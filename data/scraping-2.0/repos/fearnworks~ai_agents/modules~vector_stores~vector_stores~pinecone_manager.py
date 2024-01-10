import os
from langchain.vectorstores import Pinecone
from modules.vector_stores.embedding.openai import get_default_openai_embeddings
import pinecone 


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV') # You may need to switch with your env

class PineconeSessionManager:
    """
    A class for managing Pinecone sessions and indexes.

    Attributes:
        embeddings (OpenAIEmbeddings): The embeddings object to use for indexing.
        index_name (str): The name of the Pinecone index to use.
        index (pinecone.GRPCIndex): The Pinecone index object.
        docsearch (Pinecone): The Pinecone search object.
    """
    def __init__(self, embeddings, index_name):
        """
        Initializes a new PineconeSessionManager instance.

        Args:
            embeddings (OpenAIEmbeddings): The embeddings object to use for indexing.
            index_name (str): The name of the Pinecone index to use.
        """
        self.embeddings = embeddings
        self.index_name = index_name
        # initialize pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,  # find at app.pinecone.io
            environment=PINECONE_API_ENV  # next to api key in console
        )

        if index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(
                name=index_name,
                metric='cosine',
                dimension=len(res[0])  # 1536 dim of text-embedding-ada-002
            )
        self.index = pinecone.GRPCIndex(index_name)
        self.docsearch: Pinecone = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

def get_default_pinecone_session(index: str) -> PineconeSessionManager:
    """
    Returns a default PineconeSessionManager instance with OpenAI embeddings and a default index name.

    Returns:
        PineconeSessionManager: A new PineconeSessionManager instance.
    """
    embeddings = get_default_openai_embeddings()
    index_name = index # put in the name of your pinecone index here
    pc_session = PineconeSessionManager(embeddings, index_name)
    return pc_session