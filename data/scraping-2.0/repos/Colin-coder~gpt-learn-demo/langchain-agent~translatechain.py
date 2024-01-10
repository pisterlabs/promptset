"""Tool for the translation."""

from typing import Optional
from langchain.tools.base import BaseTool
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

class translateToChineseTool(BaseTool):

    name = "Translator"
    description = (
        "A translator from English to Chinese. "
        "Useful for when you need to translate English to Chinese."
        "Input should be the sentence that should be translated."
    )
    llm: BaseLanguageModel

    def _run(self, sentence: str) -> str:

        input = (
            "I want you to act as an English translator."
            "I will speak to you in any language and you will detect the language,"
            " translate it in English. "
            "Keep the meaning same, but make them more literary. "
            "I want you to only reply the correction, the improvements and nothing else, do not write explanations. "
            f"The sentence is `{sentence}`"
        )

        res = self.llm.predict(text = input)
        # print(f"nktest res:{res}")
        # return "translate result."
        return res
        # return NotImplementedError("translator does not support sync")
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return NotImplementedError("translator does not support async")