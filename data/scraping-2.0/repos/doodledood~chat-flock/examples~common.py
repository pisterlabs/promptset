from typing import Optional

from pathlib import Path

from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.globals import set_llm_cache
from langchain.llms.openai import OpenAI


def create_chat_model(
    model: str = "gpt-4-1106-preview",
    temperature: float = 0.0,
    cache_db_file_path: Optional[str] = "output/llm_cache.db",
) -> BaseChatModel:
    if cache_db_file_path is not None:
        Path(cache_db_file_path).parent.mkdir(parents=True, exist_ok=True)

        set_llm_cache(SQLiteCache(database_path=cache_db_file_path))

    chat_model = ChatOpenAI(temperature=temperature, model=model)

    return chat_model


def get_max_context_size(chat_model: BaseChatModel) -> Optional[int]:
    try:
        max_context_size = OpenAI.modelname_to_contextsize(chat_model.model_name)  # type: ignore
    except ValueError:
        return None

    return max_context_size
