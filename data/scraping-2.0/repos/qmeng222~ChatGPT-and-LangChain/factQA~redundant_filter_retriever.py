# module: a module is a single file containing classes and functions
# package: a package is a way of grouping related modules into a directory hierarchy (can have subpackages inside a package, forming a nested structure)

# from langchain library > embeddings package > base module, import the Embeddings class (for generating embeddings):
from langchain.embeddings.base import Embeddings

# from langchain library > vectorstores module, import the Chroma class (for storing embeddings):
from langchain.vectorstores.chroma import Chroma

# from langchain library > schema module > import the BaseRetriever class (for retrieving or fetching specific data based on certain criteria):
from langchain.schema import BaseRetriever


class RedundantFilterRetriever(BaseRetriever):
    # specify attributes to avoid hard coding:
    embeddings: Embeddings # whenever someone tries to create an instance of this class, they must provide an object that can be used to calculate embeddings
    chroma: Chroma # whenever someone tries to create an instance of this class, he must provide an already initialized instance of Chrome
    # now anyone who's using this class can control the configuration of embeddings and chrome on their own 🎉

    def get_relevant_documents(self, query):
        # note: `self` is an instance of some class
        # generate embeddings for the 'query' string:
        emb = self.embeddings.embed_query(query)

        # to find similarities to an embedding we have already calculated (`emb`) & remove duplicates automatically:
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb, # emb is an embedding or vector representation being used in the search
            lambda_mult=0.8 # lambda multiplier ranging from 0 to 1 (higher value allows for similar docs)
        )

    async def aget_relevant_documents(self):
        return []
