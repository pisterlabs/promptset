from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query: str):
        # Calculate embeddings for query
        query_embedding = self.embeddings.embed_query(query)

        # take embeddings and feed them into the ChromaDB's
        # max_marginal_relevance_search_by_vector method
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=query_embedding, lambda_mult=0.8)

    async def aget_relevant_documents(self, query: str):
        return []
