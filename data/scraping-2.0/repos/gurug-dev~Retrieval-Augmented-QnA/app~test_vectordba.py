import os
from init_setup import *
from utils import load_docs, split_docs, get_answer

import pinecone 
from langchain.vectorstores import Pinecone

documents = load_docs(directory)
docs = split_docs(documents)
try:
  pinecone.create_index(index_name, dimension=1536,
                          metric="cosine", pods=1, pod_type="p1.x1")
  index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
except:
  index = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)


query = "How to compute sample size?"  
get_answer(index, query)
