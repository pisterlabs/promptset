from langchain.llms.openai import OpenAI
import yaml
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) # read local .env file
from langchain.callbacks import StdOutCallbackHandler, WandbCallbackHandler
from datetime import datetime
from langchain.embeddings import OpenAIEmbeddings
from chromadb.api.types import Documents, Embeddings


# session_group = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
# wandb_callback = WandbCallbackHandler(
#     job_type="inference",
#     project="langchain_callback_demo",
#     group=f"minimal_{session_group}",
#     name="llm",
#     tags=["test"],
# )
# callbacks = [StdOutCallbackHandler(), wandb_callback]
#_________________________________________________________________________________________

# small_llm = OpenAI(temperature=0.0 ,frequency_penalty = 0.1 ,n = 5 ,max_tokens=1000,  model="gpt-3.5-turbo-instruct")

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0, model = 'gpt-4-1106-preview')
small_llm = ChatOpenAI(temperature=0, model = 'gpt-4-1106-preview')
# llm = OpenAI(temperature=0.00 ,frequency_penalty = 0.1 ,n = 5 ,max_tokens=1000,  model="gpt-3.5-turbo-instruct")

embedding_func = OpenAIEmbeddings()

#_________________________________________________________________________________________
def load_config(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config 

# config = load_config('backendPython/config.yaml')
#_________________________________________________________________________________________