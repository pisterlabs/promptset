from softtek_llm.embeddings import OpenAIEmbeddings
from softtek_llm.vectorStores import SupabaseVectorStore
from softtek_llm.vectorStores import Vector
from softtek_llm.chatbot import Chatbot
from softtek_llm.models import OpenAI
from softtek_llm.cache import Cache
from dotenv import load_dotenv
from Text.parse import parseText
from .save import get_embeddings_from_text, save_multiple_embeddings, save_embedding_from_text
from .search import search
import uuid
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
OPENAI_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL_NAME")
if OPENAI_CHAT_MODEL_NAME is None:
    raise ValueError("OPENAI_CHAT_MODEL_NAME not found in .env file")
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
cache = Cache(vector_store=vector_store, embeddings_model=embeddings_model)
model = OpenAI(
    api_key=OPENAI_API_KEY,
    model_name=OPENAI_CHAT_MODEL_NAME,
    api_type="azure",
    api_base=OPENAI_API_BASE,
    verbose=True,
)
chatbot = Chatbot(
    model=model,
    description="You are a text sumarizer Not a chatbot. Just Summarize text, getting the most relevant sentences, and keywords.",
    cache=None,
    verbose=True,
)

def get_text_sumary(text):
    response = chatbot.chat("Give only the summary of a text that is described by the next main sentences: " + text)
    print(response)
    return response.message.content

def get_text_keywords(text):
    response = chatbot.chat("Give only the keywords of this text: " + text)
    print(response)
    return response.message.content

def get_most_relevant_sentences(text):
    response = chatbot.chat("Give only me the most relevant sentences of the following text: " + text)
    print("RECEVING", response)
    print("RECEVING", response.message.content)
    # print("AA", response)
    return response.message.content

def get_text_analysis(text):
    response = chatbot.chat("Give me the summary, keywords and most relevant sentences of the text. Format it estrictly in a json structure like this: \{ 'summary': 'This text talks about...' \} Article:" + text)
    #print(response)
    return response.message.content

def analyze_text(text):
    print("aqui")
    main_emb = save_embedding_from_text(text)[0]["id"]
    parsed_text = parseText(text)
    print("DEBUG", parsed_text)
    embeddings = []
    relevan_sentences = []
    keywords = []

    for paragraph in parsed_text:
        embeddings.append(
            {
            "paragraph": paragraph,
            "vec": Vector(embeddings=get_embeddings_from_text(paragraph), id=str(uuid.uuid4()))
        })
        relevan_sentences.append(get_most_relevant_sentences(paragraph))
        #print("aAA", relevan_sentences)
        keywords.append(get_text_keywords(paragraph))

    ids = save_multiple_embeddings([i["vec"] for i in embeddings])
    print("[DEBUG]", relevan_sentences)
    sumary_q = get_embeddings_from_text(
        "".join(relevan_sentences)
    )

    search_res = search(query=Vector(embeddings=sumary_q), in_list=[i["id"] for i in ids])

    search_res = search_res[:5]

    paragraps = [i["paragraph"] for i in embeddings if i["vec"].id in [i.id for i in search_res]]

    sumary = get_text_sumary("".join(paragraps))

    return {
        "summary": sumary,
        "keywords": keywords,
        "relevant_sentences": relevan_sentences,
        "ids": [i["id"] for i in ids],
        "main_emb": main_emb,
    }





