import os
import openai
from dotenv import load_dotenv

load_dotenv()

# get path to the directory where this file is located
BASEDIR = os.path.abspath(os.path.dirname(__file__))
# get parent directory of BASEDIR
PARENTDIR = os.path.abspath(os.path.join(BASEDIR, os.pardir))

# Index and Hotel data
SCHEMA  = os.getenv("SCHEMA", f"{BASEDIR}/hotel_index_schema.yml")
DATADIR = os.getenv("DATADIR", f"{PARENTDIR}/data")
DATAFILE = os.getenv("DATAFILE", f"{DATADIR}/data.pkl")

# Redis information
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_ADDRESS = f"redis://{REDIS_HOST}:{REDIS_PORT}"

# AI models
openai.api_key = os.getenv('OPENAI_API_KEY')
CHAT_MODEL = os.getenv('OPENAI_CHAT_MODEL')
VECTORIZER = os.getenv('HF_VECTOR_MODEL', 'all-MiniLM-L6-v2')