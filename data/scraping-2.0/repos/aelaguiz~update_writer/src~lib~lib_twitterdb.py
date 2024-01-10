import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from . import lib_logging

# Assuming docdb is a Chroma instance or similar
docdb = None
llm = None

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
DOCBDB_PATH = None
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE"))

COMPANY_ENV = None

def set_company_environment(company_env):
    print(f"Setting company environment to {company_env}")
    global COMPANY_ENV
    global DOCBDB_PATH

    #TWITTER_DOCDB_PATH="twitter_db"
    DOCBDB_PATH = os.getenv(f'{company_env}_DOCDB_PATH')
    COMPANY_ENV = company_env

    print(f"DOCBDB_PATH: {DOCBDB_PATH} - COMPANY_ENV: {COMPANY_ENV}_DOCBDB_PATH")

def get_embedding_fn():
    return OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'), timeout=30)

def get_llm():
    global llm 

    if not llm:
        llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)

    return llm

def get_docdb():
    global docdb

    if not docdb:
        docdb = Chroma(embedding_function=get_embedding_fn(), persist_directory=DOCBDB_PATH)
        print(f"Loaded docdb from {DOCBDB_PATH}")

    return docdb

def get_tweet_ids():
    docdb = get_docdb()
    res = docdb.get(include=['metadatas'])
    for id, metadata in zip(res['ids'], res['metadatas']):
        print(id, metadata)
        yield metadata['tweet_id']

def add_tweet(tweet_details):
    logger = lib_logging.get_logger()
    try:
        # Assuming tweet_details has 'id' and 'text' keys
        metadata = {
            'type': 'tweet',
            'tweet_id': tweet_details['id'],
            'tweet_text': tweet_details['text']
        }
        docdb.add_texts([tweet_details['text']], metadatas=[metadata])
    except Exception as e:
        logger.error(f"Error loading tweet into Chroma: {e}")
        raise RuntimeError(f"Error loading tweet into Chroma: {e}")