from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from typing import List


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(
        self,
        query: str,
    ) -> List[Document]:
        # Calculate emebddings for the query
        embedding = self.embeddings.embed_query(query)

        # Search for similar documents
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=embedding, lambda_mult=0.8
        )

    async def aget_relevant_documents(self) -> List[Document]:
        return []
