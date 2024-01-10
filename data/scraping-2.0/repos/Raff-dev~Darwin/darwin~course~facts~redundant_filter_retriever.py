from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain.vectorstores.chroma import Chroma


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def _get_relevant_documents(self, query, *_, **__):
        emb = self.embeddings.embed_query(query)
        result = self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8,
        )
        return result
