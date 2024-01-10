__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2022, 23. All rights reserved."

from langchain.agents import AgentExecutor
from langchain.cache import InMemoryCache, SQLiteCache, BaseCache
from typing import AnyStr
from langchain.chat_models import ChatOpenAI


class LLMBaseAgent(object):
    in_memory_cache = "in_memory_cache"
    sql_lite_cache = "sql_lite_cache"
    no_cache = ""

    def __init__(self, chat_handle: ChatOpenAI, agent: AgentExecutor, cache_model: AnyStr):
        """
        Default constructor for LangChain agent to be used with ChatGPT
        :param chat_handle Handle or reference to the large language model
        :param agent LangChain agent executor
        :param cache_model Cache model (in memory, SQL lite,...)
        """
        assert cache_model in [LLMBaseAgent.in_memory_cache, LLMBaseAgent.sql_lite_cache, LLMBaseAgent.no_cache], \
            f'Cache model {cache_model} is not supported'

        self.chat_handle = chat_handle
        self.agent = agent
        if cache_model:
            self.chat_handle.llm_cache = LLMBaseAgent.set_cache(cache_model)

    @staticmethod
    def set_cache(cache_model: AnyStr) -> BaseCache:
        """
        Initialize the cache for this class and any other class
        @param cache_model: Supported cache model
        @return: Cache structure if cache model is supported, a NotEmplementedError otherwise
        """
        if cache_model == LLMBaseAgent.in_memory_cache:
            return InMemoryCache()
        elif cache_model == LLMBaseAgent.sql_lite_cache:
            return SQLiteCache(database_path=".langchain.db")
        else:
            raise NotImplementedError(f'Cache model {cache_model} is not supported')

    def __call__(self, input_prompt_str: AnyStr) -> AnyStr:
        """
            :param input_prompt_str Input stream
            :return answer from ChatGPT
        """
        return self.agent.run(input_prompt_str)
