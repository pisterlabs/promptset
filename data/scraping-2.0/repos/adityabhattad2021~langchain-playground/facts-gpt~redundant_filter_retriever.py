from typing import List
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.schema import BaseRetriever, Document


class RedundantFilterRetriever(BaseRetriever):

    embeddings: Embeddings
    vectorDB: VectorStore

    def get_relevant_documents(self, query: str) -> List[Document]:
        emb = self.embeddings.embed_query(query)
        res = self.vectorDB.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8,
        )
        return res
    

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return []