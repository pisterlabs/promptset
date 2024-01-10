import faiss
from langchain.vectorstores import FAISS

import pickle


def store_embeddings(docs, embeddings, store_name, path):

  vectorStore = FAISS.from_documents(docs, embeddings)
  with open(f"{path}/faiss_{store_name}.pkl", "wb") as f:
    pickle.dump(vectorStore, f)

def load_embeddings(store_name, path):
  with open(f"{path}/faiss_{store_name}.pkl", "rb") as f:
    VectorStore = pickle.load(f)

  return VectorStore