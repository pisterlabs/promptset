from openai.embeddings_utils import get_embedding, get_embeddings
from openai.embeddings_utils import cosine_similarity
import openai

openai.api_key = "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"

def get_openai_embedding(text):
    embedding = get_embedding(text, engine="text-embedding-ada-002")
    return embedding

def get_similarity(embed1, embed2):
    similarity = cosine_similarity(embed1, embed2)
    return similarity
