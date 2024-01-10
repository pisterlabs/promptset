from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone
import os
os.getenv("OPENAI_API_KEY")


class PineconeIndex():

    def create_pinecone_index_new(nodes):
        pinecone.init(environment="gcp-starter",
                      api_key="775632e7-fba1-4a9a-9ea1-9c0f170bb08a")
        index_name = "test1"
        # vector_store = pinecone_index.as_vector_store()
        pinecone_index = pinecone.Index(index_name=index_name)
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        embed_model = OpenAIEmbedding(
            model='text-embedding-ada-002')
        service_context = ServiceContext.from_defaults(embed_model=embed_model)
        index = VectorStoreIndex(
            nodes=nodes,
            service_context=service_context,
            storage_context=storage_context,
        )
        # Methods to make the embedding and indexing process more efficient
        # Method 1 : create a function to do a similarity search for each node embedding against the already existing vectors in pinecone index, assign it a cosine similarity score, and add it to the index only if the score is greater than 0.95.
        # Method 2:

        return index

    def use_existing_pinecone_index():
        pinecone.init(environment="gcp-starter",
                      api_key="775632e7-fba1-4a9a-9ea1-9c0f170bb08a")
        index_name = "test1"
        pinecone_index = pinecone.Index(index_name=index_name)
        vector_store = PineconeVectorStore(pinecone_index)
        index = VectorStoreIndex.from_vector_store(vector_store)
        return index
