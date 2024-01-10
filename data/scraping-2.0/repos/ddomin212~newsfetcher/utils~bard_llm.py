import os
from typing import Any, List, Mapping, Optional

from bardapi import BardCookies
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

os.environ["CONVERSATION_ID"] = ""


class BardLLM(LLM):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        load_dotenv()
        print(prompt)
        bard = BardCookies(
            token_from_browser=True,
            conversation_id=os.environ["CONVERSATION_ID"],
        )
        response = bard.get_answer(prompt)
        if not len(os.environ["CONVERSATION_ID"]) > 0:
            os.environ["CONVERSATION_ID"] = response["conversation_id"]
        return response["content"]
