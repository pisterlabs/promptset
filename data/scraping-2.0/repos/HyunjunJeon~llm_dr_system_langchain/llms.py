from bardapi import Bard
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

from load_env import load_bard_env

load_bard_env()


# https://python.langchain.com/docs/modules/model_io/llms/custom_llm
class BardLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "Bard_PALM2_20231207"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        response = Bard(
            timeout=60,
        ).get_answer(
            prompt
        )["content"]
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


"""
사용할 LLM 정의
"""


def get_bard() -> BardLLM:
    return BardLLM()


def get_openai(
    temperature: float = 0.1,
    model_name: str = "gpt-3.5-turbo-1106",
    streaming: bool = False,
    callbacks: List[BaseCallbackHandler] = [],
) -> ChatOpenAI:
    return ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        streaming=streaming,
        callbacks=callbacks,
    )
