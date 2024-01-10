import os
from pathlib import Path

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Redis

from parser.pdf import pdf_to_text
from core.chunkers import chunk_text_file


class Settings(object):
    BASE_DIR = Path(__file__).parent.parent  # nifty absolute project path
    ABS_BASE_DIR = BASE_DIR.parent  # project absolute path

    DATA_DIR = BASE_DIR / 'data'
    INPUT_DATA_DIR = DATA_DIR / 'input'
    OUTPUT_DATA_DIR = DATA_DIR / 'output'

    if not OUTPUT_DATA_DIR.exists():
        os.mkdir(OUTPUT_DATA_DIR)

    INPUT_NIFTY_BRIDGE_FILEPATH = INPUT_DATA_DIR / os.environ['INPUT_NIFTY_BRIDGE_FILENAME']
    OUTPUT_NIFTY_BRIDGE_FILEPATH = OUTPUT_DATA_DIR / os.environ['OUTPUT_NIFTY_BRIDGE_FILENAME']

    if not INPUT_NIFTY_BRIDGE_FILEPATH.exists():
        raise FileNotFoundError(f'INPUT_NIFTY_BRIDGE_FILEPATH filepath `{INPUT_NIFTY_BRIDGE_FILEPATH}` does not exists')

    WEBAPP_HOST = os.environ['WEBAPP_HOST']
    WEBAPP_PORT = os.environ['WEBAPP_PORT']

    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_MAX_TOKENS = int(os.environ["OPENAI_MAX_TOKENS"])
    OPENAI_USE_MODEL = 'gpt-3.5-turbo'

    REDIS = {
        'NAME': os.environ['REDIS_DB'],
        'HOST': os.environ['REDIS_HOST'],
        'PORT': os.environ['REDIS_PORT'],
    }
    REDIS_URL = f"redis://{REDIS['HOST']}:{REDIS['PORT']}"

    X_API_KEY_TOKEN = os.environ['X_API_KEY_TOKEN']
    AUTH_HEADER = 'x_api_key_token'  # only lower case

    # Initialize text chunks
    if not OUTPUT_NIFTY_BRIDGE_FILEPATH.exists():
        pdf_to_text(INPUT_NIFTY_BRIDGE_FILEPATH, OUTPUT_NIFTY_BRIDGE_FILEPATH)
    TEXT_CHUNKS = chunk_text_file(OUTPUT_NIFTY_BRIDGE_FILEPATH)
    EMBEDDINGS_REDIS, _ = Redis.from_texts_return_keys(
        texts=TEXT_CHUNKS,
        embedding=OpenAIEmbeddings(),
        redis_url=REDIS_URL,
        index_name="link"
    )


settings = Settings()
