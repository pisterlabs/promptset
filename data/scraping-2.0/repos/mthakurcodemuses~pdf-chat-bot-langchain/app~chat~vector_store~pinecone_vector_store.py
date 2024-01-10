import os

import pinecone
from langchain.vectorstores import Pinecone

from app.chat.embeddings.open_ai_embeddings import open_ai_embeddings

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENV_NAME"))

pinecone_vector_store = Pinecone.from_existing_index(os.getenv("PINECONE_INDEX_NAME"), open_ai_embeddings)

def build_retriever(chat_args, doc_search_limit):
    search_kwargs = {"filter":{"pdf_id": chat_args.pdf_id}, "k": doc_search_limit}
    return pinecone_vector_store.as_retriever(search_kwargs=search_kwargs)
