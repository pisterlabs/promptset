from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
  embeddings: Embeddings
  chroma: Chroma

  def get_relevant_documents(self, query):
      # Calculate embeddings for the query
      emb = self.embeddings.embed_query(query)


      #Take embeddings and feed them into the retriever 
      # with max_marginal_relavance_search_by_vector
      return self.chroma.max_marginal_relevance_search_by_vector(
         embedding=emb,
         lambda_mult=0.8, #controls how much the retriever should prioritize
                          # diversity over relevance, higher the value, higher the similarity
      )
  
  async def age_relevant_documents(self):
     return []