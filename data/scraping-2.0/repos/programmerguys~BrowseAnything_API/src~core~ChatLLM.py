import os
from typing import Union, List
from uuid import uuid4

import graphsignal
from dotenv import load_dotenv
from langchain import ConversationChain
from langchain.callbacks import PromptLayerCallbackHandler
from langchain.memory import ZepMemory

from src.core.Tools import load_llm, parse_roleMessages_to_prompts
from src.entity.QueryForm import RoleMessage

load_dotenv()


class ChatLLM(object):
    def __init__(self):
        # Set this to your Zep server URL
        self.zep_api_url = os.environ.get('ZEP_API_URL')
        self.zep_api_key = os.environ.get('ZEP_API_KEY')

        # 配置llm模型
        self._temperature = float(os.environ.get('OPENAI_TEMPERATURE'))
        self._model_name = os.environ.get('OPENAI_MODEL_NAME')
        self._openai_api_key = os.environ.get('OPENAI_API_KEY')
        self._openai_proxy = os.environ.get('OPENAI_PROXY')
        self._openai_api_base = os.environ.get('OPENAI_API_BASE')
        self._max_tokens = int(os.environ.get('OPENAI_MAX_TOKENS'))
        self.llm = load_llm(
            self._temperature,
            self._model_name,
            self._openai_api_key,
            self._openai_proxy,
            self._openai_api_base,
            self._max_tokens
        )
        self.graphsignal_name = os.environ.get('GRAPHSIGNAL_NAME')

        graphsignal.configure(api_key='16ddccf7551cb0e1b4a20435c495e999', deployment=self.graphsignal_name)

    def get_memory(self, session_id: str = None):
        if session_id is None:
            session_id = str(uuid4())

        print(f"session_id: {session_id}")

        return ZepMemory(
            session_id=session_id,
            url=self.zep_api_url,
            api_key=self.zep_api_key,
            input_key="input",
            memory_key="chat_history",
        )

    def get_chain(
            self,
            temperature: float = None,
            model_name: str = None,
            roleMessages: Union[List[RoleMessage], None] = None,
            pl_tags=None,
            memory: str = None,
    ):
        callbacks = []
        if pl_tags is not None:
            promptLayer = PromptLayerCallbackHandler(pl_tags=pl_tags)
            callbacks.append(promptLayer)

        _llm = load_llm(temperature, model_name, self._openai_api_key, self._openai_proxy, self._openai_api_base,
                        self._max_tokens, callbacks)

        text_qa_template = parse_roleMessages_to_prompts(roleMessages, True)
        return ConversationChain(llm=_llm, verbose=True, prompt=text_qa_template, memory=memory)


llms = ChatLLM()
