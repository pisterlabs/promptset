# Pinecone vectorstore initialization

import pinecone
from langchain.vectorstores import Pinecone

def create_pinecone_vectordb(index_name, embed_function):
    index = pinecone.Index(index_name)  # Switch back to normal index for langchain
    vectordb = Pinecone(index, embed_function, "text")  # Creates vectorstoredb
    print("PineconeDB created")
    return vectordb