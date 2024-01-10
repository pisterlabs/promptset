import logging
import time

import tiktoken
from django.conf import settings
from langchain.callbacks import get_openai_callback
from langchain.chat_models import AzureChatOpenAI
from langchain.schema.output_parser import StrOutputParser

logger = logging.getLogger(__name__)


def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_llm_response(prompt, prompt_kwargs_dict):
    model = AzureChatOpenAI(
        openai_api_type=settings.OPENAI_API_TYPE,
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_version=settings.OPENAI_API_VERSION,
        openai_api_base=settings.OPENAI_API_BASE,
        deployment_name=settings.OPENAI_API_DEPLOYMENT_NAME,
        temperature=settings.OPENAI_API_TEMPERATURE,
    )
    chain = prompt | model | StrOutputParser()
    start_time = time.time()
    with get_openai_callback() as cb:
        resp = chain.invoke(prompt_kwargs_dict)
    end_time = time.time()
    execution_time = end_time - start_time
    return resp, cb, execution_time
