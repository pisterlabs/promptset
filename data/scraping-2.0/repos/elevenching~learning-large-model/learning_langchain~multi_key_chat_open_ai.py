import os
from typing import Any, List, Optional

from dotenv import load_dotenv, find_dotenv
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, ChatResult


# 实现多个OpenAI API Key 轮询使用，避免触碰 limit
def round_robin(keys):
    while True:
        for key in keys:
            yield key


class MultiKeyChatOpenAI(ChatOpenAI):
    keys: List[str] = []

    def __init__(self, **kwargs):
        # 取出 keys 参数，ChatOpenAI 要检查参数，因此需要先取出
        keys = kwargs.pop('keys', [])

        # 加载 .env 文件，获取 OPENAI_API_KEY，初始化 ChatOpenAI 需要，否则传入 openai_api_key
        load_dotenv(find_dotenv())
        # 设置本地代理 openai_proxy 或代理服务器地址 openai_api_base=https://api.openai-proxy.com
        kwargs.setdefault('openai_proxy', os.environ['OPENAI_LOCAL_PROXY'])
        kwargs.setdefault('model', 'gpt-3.5-turbo-16k-0613')
        kwargs.setdefault('verbose', True)
        super().__init__(**kwargs)

        if len(keys) == 0:
            keys = os.environ['OPENAI_API_KEYS'].split(',')
        self.__dict__['robin'] = round_robin(keys)

    def _generate_with_cache(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        self.openai_api_key = next(self.__dict__['robin'])
        return super()._generate_with_cache(messages, stop, run_manager, **kwargs)

    async def _agenerate_with_cache(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        self.openai_api_key = next(self.__dict__['robin'])
        return await super()._agenerate_with_cache(messages, stop, run_manager, **kwargs)
