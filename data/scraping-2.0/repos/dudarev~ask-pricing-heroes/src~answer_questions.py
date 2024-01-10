import os
from pathlib import Path

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from cache import Cache

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


CACHE_DIR = Path(__file__).parent.parent / "data/cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
CACHE_FILE = CACHE_DIR / "cache.json"


def get_chain(api_key=None):
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        CHROMA_DATA_DIR = Path(__file__).parent.parent / "data/chroma"
        if not CHROMA_DATA_DIR.exists():
            raise Exception(
                "Please run create_embeddings.py first to create the vectorstore. "
                "See README.md for more details."
            )
        vectordb = Chroma(
            persist_directory=str(CHROMA_DATA_DIR),
            embedding_function=OpenAIEmbeddings(openai_api_key=api_key),
        )
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            ChatOpenAI(
                temperature=0, model_name="gpt-3.5-turbo", openai_api_key=api_key
            ),
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
        )
    else:
        vectordb = None
        chain = None
    return chain


cache = Cache(CACHE_FILE)


def answer_question(question, api_key=None):
    if cached := cache.get(question):
        return cached
    chain = get_chain(api_key=api_key)
    if chain:
        res = chain({"question": question}, return_only_outputs=True)
        cache.set(question, res)
    else:
        res = {
            "answer": "Please set 'OpenAI API Key' in the sidebar to get non-cached answers."
        }
    return res
