#!/usr/bin/python3

#  Copyright (c) 2023 Paradigma Digital S.L.

#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from enum import Enum
from typing import Any
from typing import List
from typing import Dict
from typing import Mapping
from typing import Optional

from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.embeddings import FakeEmbeddings

from .base import BaseModel


class FakeStaticLLM(LLM):

    """
    Fake Static LLM wrapper for testing purposes.
    """

    response: str = "positive"

    @property
    def _llm_type(
        self,
    ) -> str:
        """
        Return type of llm.
        """

        return "fake-static"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        Return static response.
        """

        return self.response

    @property
    def _identifying_params(
        self,
    ) -> Mapping[str, Any]:
        return {}

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self._call(prompt, stop, run_manager, **kwargs)


class FakePromptCopyLLM(LLM):

    """
    Fake Prompt Copy LLM wrapper for testing purposes.
    """

    @property
    def _llm_type(
        self,
    ) -> str:
        """
        Return type of llm.
        """

        return "fake-prompt-copy"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        Return prompt.
        """

        return prompt

    @property
    def _identifying_params(
        self,
    ) -> Mapping[str, Any]:
        return {}

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self._call(prompt, stop, run_manager, **kwargs)


class FakeListLLM(LLM):

    """
    Fake LLM wrapper for testing purposes.
    """

    responses: List = ["uno", "dos", "tres"]
    i: int = 0

    @property
    def _llm_type(
        self,
    ) -> str:
        """
        Return type of llm.
        """

        return "fake-list"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        First try to lookup in queries, else return 'foo' or 'bar'.
        """

        response = self.responses[self.i]
        self.i += 1

        return response

    @property
    def _identifying_params(
        self,
    ) -> Mapping[str, Any]:
        return {}

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self._call(prompt, stop, run_manager, **kwargs)


class ModelTypes(Enum):

    """
    FakeLLM Model Types.
    """

    MODEL_1: str = "fake-static"
    MODEL_2: str = "fake-prompt_copy"
    MODEL_3: str = "fake-list"


class FakeLLM(BaseModel):
    """
    Fake LLM class.
    """

    LLM_MAPPING: Dict[str, LLM] = {
        ModelTypes.MODEL_1.value: FakeStaticLLM,
        ModelTypes.MODEL_2.value: FakePromptCopyLLM,
        ModelTypes.MODEL_3.value: FakeListLLM,
    }

    def __init__(
        self,
        model_name: Optional[str] = "",
        model_params: Optional[Dict] = None,
        model_provider_token: Optional[str] = "",
    ) -> None:
        super(FakeLLM, self).__init__()
        self.model_params = model_params
        self.model_provider_token = model_provider_token

        self._embeddings = FakeEmbeddings(size=64)
        if model_name in self.LLM_MAPPING:
            self._llm = self.LLM_MAPPING[model_name]()
        else:
            raise ValueError(
                f"{self.__class__.__name__} error creating object. "
                f"{model_name} is not in the list of supported FakeLLMS: "
                f"{[i.value for i in ModelTypes]}"
            )
