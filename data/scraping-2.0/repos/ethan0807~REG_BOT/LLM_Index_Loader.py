from llama_index import LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
import openai
import Globals

# Loads the vector index from the storage directory

globals = Globals.Defaults()
openai.api_key = globals.open_api_key
default_model = globals.default_model
default_temperature = globals.default_temperature
default_max_chunk_size = globals.default_max_chunk_size
index_path = globals.index_path


def load_index():
    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(
        temperature=default_temperature, model_name=default_model))

    # configure service context
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size=default_max_chunk_size)

    # rebuild storage context
    storage_context = StorageContext.from_defaults(
        persist_dir=index_path)

    # load index
    return load_index_from_storage(storage_context)
