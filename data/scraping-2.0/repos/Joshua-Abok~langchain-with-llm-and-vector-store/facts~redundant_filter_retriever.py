from typing import Any, Dict, List, Optional
from langchain.callbacks.manager import Callbacks
from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever

# class extends BaseRetriever
class RedundantFilterRetriever(BaseRetriever):
        #  create our attributes embeddings & chroma 
    embeddings: Embeddings
    chroma: Chroma          # provide already initialized instance of chroma 

    # Now define relevant funct. 
    def get_relevant_documents(self, query):
        # calculate embeddings for the 'query' string 
        emb = self.embeddings.embed_query(query)

        # take embeddings & feed them into the max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,        # baseline emb to find relevant docs for 
            lambda_mult=0.8       # tolerance for similar docs
        )
    
    async def aget_relevant_documents(self):
        return []