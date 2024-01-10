from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
  embeddings: Embeddings
  chroma: Chroma 

  def get_relevant_documents(self, query):
    # calculate embeddings for the query string
    emb = self.embeddings.embed_query(query)

    # feed embeddings into max_marginal_relevance_search_by_vector
    return self.chroma.max_marginal_relevance_search_by_vector(
      embedding=emb,
      lambda_mult=0.8
    )

    return []

  async def aget_relevant_documents(self):
    # not implemented
    return []