## for embeddings
from langchain.embeddings import HuggingFaceEmbeddings

def setup_embeddings():
    # Huggingface embedding setup
    print(">> Prep. Huggingface embedding setup")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    return HuggingFaceEmbeddings(model_name=model_name)