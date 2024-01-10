"""Main entrypoint for the app."""
import os
from typing import Optional
from timeit import default_timer as timer
from dotenv import find_dotenv, load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from app_modules.llm_loader import LLMLoader
import datetime

from app_modules.utils import get_device_types, init_settings

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

if os.environ.get("LANGCHAIN_DEBUG") == "true":
    import langchain

    langchain.debug = True

# Constants
init_settings()


def app_init(custom_handler: Optional[BaseCallbackHandler] = None):
    start = timer()

    print(f"App init started at {datetime.datetime.now()}")

    llm_model_type = os.environ.get("LLM_MODEL_TYPE")
    n_threds = int(os.environ.get("NUMBER_OF_CPU_CORES") or "4")

    hf_embeddings_device_type, hf_pipeline_device_type = get_device_types()
    print(f"hf_embeddings_device_type: {hf_embeddings_device_type}")
    print(f"hf_pipeline_device_type: {hf_pipeline_device_type}")

    llm_loader = LLMLoader(llm_model_type)
    llm_loader.init(
        n_threds=n_threds,
        hf_pipeline_device_type=hf_pipeline_device_type,
        custom_handler=custom_handler,
    )
    end = timer()
    print(f"App init completed in {end - start:.3f}s")

    return llm_loader
