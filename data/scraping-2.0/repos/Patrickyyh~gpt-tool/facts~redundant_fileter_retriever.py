from typing import Any, Dict, List, Optional
from langchain.callbacks.manager import Callbacks
from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever
from langchain.embeddings import OpenAIEmbeddings

class RedudantFilterRetriever(BaseRetriever):

    ## Please provide an already initialized Embeddings
    embeddings: Embeddings

    ## Please provide an already initialized Chroma DB
    chroma: Chroma

    def get_relevant_documents(self, query):

        # Calculate embedings for user's query
        emb = self.embeddings.embed_query(query)

        # take embeddingsand feed them into the max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding = emb,
            lambda_mult = 0.5)

    async def get_relevant_documents_async(self, query):
        return []



