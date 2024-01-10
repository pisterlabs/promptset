from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import chromadb
from tqdm import tqdm

def retrieve_chromadb(collection_name: str="one_piece"):
    persistent_client = chromadb.PersistentClient(path=f"data/chroma/{collection_name}/")
    vectordb = Chroma(
        collection_name=collection_name,
        client=persistent_client,
        embedding_function=OpenAIEmbeddings(),
        )
    return vectordb
