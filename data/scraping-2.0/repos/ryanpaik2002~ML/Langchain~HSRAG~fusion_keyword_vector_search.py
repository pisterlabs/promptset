

""" 
Hybrid Keyword/Vector Search Snippet

Source: https://medium.com/towards-data-science/-pipelines-with-hybrid-search-c75203c2f2f5

Uses Weaviate, 

TODO: 
    - Replace Weaviate with Faiss
    - Run with OpenaAI API and Local Llama Server
    - secondary cache, then store DB with Redis

"""




hybrid_score = (1-alpha) * sparse_score + alpha * dense_score
"""
alpha = 1 -> pure vector search
alaph = 0 -> pure keyword search
"""

# Define and populate vector store
# See details here https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2
vectorstore = ...

# Set vectorstore as retriever
retriever = vectorstore.as_retriever()


from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever

retriever = WeaviateHybridSearchRetriever(
    alpha = 0.5,               # defaults to 0.5, which is equal weighting between keyword and semantic search
    client = client,           # keyword arguments to pass to the Weaviate client
    index_name = "LangChain",  # The name of the index to use
    text_key = "text",         # The name of the text key to use
    attributes = [],           # The attributes to return in the results
)



""" rest of the rag pipeline remains"""
