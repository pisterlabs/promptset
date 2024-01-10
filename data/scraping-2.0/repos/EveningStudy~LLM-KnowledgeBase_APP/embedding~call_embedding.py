import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llm.call_llm import parse_llm_api_key

def get_embedding(embedding: str, embedding_key: str=None, env_file: str=None):
    if embedding_key == None:
        embedding_key = parse_llm_api_key(embedding)
    # elif embedding == 'm3e':
    return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    # else:
    #     raise ValueError(f"embedding {embedding} not support ")
