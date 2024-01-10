from softtek_llm.embeddings import OpenAIEmbeddings
from softtek_llm.vectorStores import SupabaseVectorStore
from softtek_llm.vectorStores import Vector
from Text.parse import parseText
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
if not SUPABASE_API_KEY:
    raise ValueError("SUPABASE_API_KEY is not set")
SUPABASE_URL = os.getenv("SUPABASE_URL")
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL is not set")
SUPABASE_INDEX_NAME = os.getenv("SUPABASE_INDEX_NAME")
if not SUPABASE_INDEX_NAME:
    raise ValueError("SUPABASE_INDEX_NAME is not set")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME")
if not OPENAI_EMBEDDINGS_MODEL_NAME:
    raise ValueError("OPENAI_EMBEDDING_MODEL_NAME is not set")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
if not OPENAI_API_BASE:
    raise ValueError("OPENAI_API_BASE is not set")
embeddings_model = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model_name=OPENAI_EMBEDDINGS_MODEL_NAME,
    api_type="azure",
    api_base=OPENAI_API_BASE,
)
vector_store = SupabaseVectorStore(
    api_key=SUPABASE_API_KEY,
    url=SUPABASE_URL,
    index_name=SUPABASE_INDEX_NAME,
)
def get_embeddings_from_text(text):
    return embeddings_model.embed(text)

def save_multiple_embeddings(list):
    res = vector_store.add(vectors=list)
    ids = [{"id": i["id"]} for i in res]
    return ids

def save_text(text):
    paragraphs = parseText(text)
    embeddings = []
    for paragraph in paragraphs:
        embeddings.append(Vector(embeddings=get_embeddings_from_text(paragraph)))
    res = vector_store.add(vectors=embeddings)
    ids = [{"id": i["id"]} for i in res]
    return ids

def save_embedding_from_text(text, id = None):
    emb = get_embeddings_from_text(text)
    new_vector = Vector(embeddings=emb, id=id)
    res = vector_store.add(vectors=[new_vector])
    print(res)
    return [{"id": i["id"]} for i in res]

def get_embeddings_from_bigtext(text):
    paragraphs = parseText(text)
    embeddings = []
    for paragraph in paragraphs:
        embeddings.append(Vector(embeddings=get_embeddings_from_text(paragraph)))
    
    return embeddings


