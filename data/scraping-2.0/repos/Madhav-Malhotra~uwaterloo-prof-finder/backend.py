"""BACKEND.PY
This script finds Uwaterloo faculty members with bios that match your query
Dependencies: `dotenv`, `langchain`, `faiss-cpu`
"""

# Load libraries
import os
import argparse
from dotenv import load_dotenv
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings

# Load API key
load_dotenv(".env")
KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def load_embeddings(key: str) -> HuggingFaceInferenceAPIEmbeddings:
    """ Load the embeddings model """
    if key is None:
        raise ValueError("Please pass HuggingFace API key")

    return HuggingFaceInferenceAPIEmbeddings(
        api_key=key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

def load_db(embeddings: HuggingFaceInferenceAPIEmbeddings, 
            db_path: str="data/bios_faiss_merged") -> FAISS:
    """ Load the database from disk """
    
    # Load individual vectorstores
    base = FAISS.load_local("data/bios_faiss_0", embeddings)
    others = [FAISS.load_local(f"data/bios_faiss_{i}", embeddings) for i in range(400, 10800, 400)]

    # Merge vectorstores
    for other in others:
        base.merge_from(other)
    return base

def search(query: str, db: FAISS, k: int=4) -> list:
    """ Search for the top k similar records in the database """
    return db.similarity_search(query, k=k)

def main(query: str, k: int=4):
    # Load data
    model = load_embeddings(KEY)
    db = load_db(model)

    # Search
    results = search(query, db, k=k)
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="A topic you're interested in")
    parser.add_argument("-r", "--results", help="Number of results to return", type=int, default=4)

    args = parser.parse_args()
    main(args.query, args.results)