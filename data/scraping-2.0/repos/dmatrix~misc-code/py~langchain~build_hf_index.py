import os
import time

from pathlib import Path

from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma 

import ray

CHUNKS = 100

def split_pages(pages, n): 
    for i in range(0, len(pages), n):
        yield pages[i:i + n]

@ray.remote
def create_embeddings(shard: str, vector_db_path: str, hf_ai_key: str):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_ai_key
    
    embeddings = HuggingFaceEmbeddings()
    persist_dir = f"{vector_db_path}"
    vectordb = Chroma.from_documents(shard, embedding=embeddings,
                                 persist_directory=persist_dir)
    vectordb.persist()
    
    return len(shard)


def create_index(document: str, vector_db_path: str, hf_ai_key: str) -> None:

    loader = PyPDFLoader(document)
    pages = loader.load_and_split()
    # chunks = list(split_pages(pages, CHUNKS))

    # We could use Ray Distribute creating indexes but as is fast enough
    vectordbs = ray.get(create_embeddings.remote(pages, vector_db_path, hf_ai_key))
    print(f"Embeddings created from {vectordbs} pages")

if __name__ == "__main__":
    KEY = "your-key"
    document = Path(Path.cwd(), "hai_ai_index_report.pdf").as_posix()
    vector_db_path = Path(Path.cwd(), "vector_hf_db").as_posix()
    index_path = Path(vector_db_path, "index")
    
    start = time.time()
    create_index(document=document, vector_db_path=vector_db_path, hf_ai_key=KEY)
    end = time.time()
    print(f"Time to build index: {end-start:.2f} seconds")